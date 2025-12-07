import torch
from typing import List, Tuple, Optional, Dict, Any
from transformers.cache_utils import Cache

class VectorizedDynamicCache(Cache):
    """
    Optimization Level 1: Vectorized Indexing.
    
    This class replaces the slow Python dictionary lookups in the original
    'get_transfer_cache' with GPU-native scatter/gather operations.
    """
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.num_hidden_layers = num_hidden_layers
        self._transfer_order = None

    def get_transfer_cache(self, layer_idx, cache_position, prv_cache_position):
        # Calculate order once per step (at layer 0) and reuse
        if layer_idx > 0:
            if self._transfer_order is None:
                 # Fallback safety if layer 0 wasn't called first
                 return self._compute_transfer_order(cache_position, prv_cache_position)
            
            order = self._transfer_order
            if layer_idx == self.num_hidden_layers - 1:
                self._transfer_order = None # Reset for next step
            return order
        
        return self._compute_transfer_order(cache_position, prv_cache_position)

    def _compute_transfer_order(self, cache_position, prv_cache_position):
        """
        Computes the reordering indices using GPU vectorization instead of CPU loops.
        """
        B = cache_position.shape[0]
        device = cache_position.device

        # 1. Identify indices for the Current State (Old Cache + New Tokens)
        # Note: We flatten the nonzero results to compatible shapes
        prev_indices = prv_cache_position.nonzero(as_tuple=True)[1].view(B, -1)
        # The 'new' tokens are those NOT in the previous cache
        new_indices = (~prv_cache_position).nonzero(as_tuple=True)[1].view(B, -1)
        
        # current_order: [B, Seq_Len_Current]
        current_order = torch.cat([new_indices, prev_indices], dim=-1)

        # 2. Identify indices for the Next State (Target)
        keep_indices = cache_position.nonzero(as_tuple=True)[1].view(B, -1)
        drop_indices = (~cache_position).nonzero(as_tuple=True)[1].view(B, -1)
        
        # next_order: [B, Seq_Len_Total] - The values we WANT to find in current_order
        next_order = torch.cat([drop_indices, keep_indices], dim=-1)

        # 3. Vectorized "Value to Index" Mapping
        # Instead of a dict, we use a scatter operation to create a lookup table.
        # We assume indices are within a reasonable range (Sequence Length).
        
        max_val = int(max(current_order.max(), next_order.max()).item()) + 1
        
        # Create a lookup table initialized with a dummy value (e.g., -1)
        # lookup_table shape: [B, Max_Token_ID]
        lookup_table = torch.full((B, max_val), -1, dtype=torch.long, device=device)
        
        # The values in 'current_order' are the keys. 
        # The indices (0, 1, 2...) are the values we want to retrieve.
        src_indices = torch.arange(current_order.shape[1], device=device).expand(B, -1)
        
        # Scatter: lookup_table[b, current_order[b, i]] = i
        lookup_table.scatter_(1, current_order, src_indices)

        # 4. Gather the Transfer Order
        # We look up the values from 'next_order' in our table to get their
        # index positions in 'current_order'.
        transfer_order = torch.gather(lookup_table, 1, next_order)

        self._transfer_order = transfer_order
        return transfer_order

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # (Standard implementation using the optimized get_transfer_cache)
        cache_kwargs = cache_kwargs or {}
        B, n_h, _, h_d = key_states.shape
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        cache_position = cache_kwargs.get("cache_position", None)

        if prv_cache_position is None:
            # Initialization phase (Standard append)
            # Make sure we're initializing an empty cache
            assert len(self.key_cache) <= layer_idx, "Cache should be empty during initialization"
            
            key_states_selected = key_states[cache_position[:, None, :, None].expand_as(key_states)].view(B, n_h, -1, h_d)
            self.key_cache.append(key_states_selected)
            value_states_selected = value_states[cache_position[:, None, :, None].expand_as(value_states)].view(B, n_h, -1, h_d)
            self.value_cache.append(value_states_selected)
        else:
            # Reorder phase
            transfer_order = self.get_transfer_cache(layer_idx, cache_position, prv_cache_position)
            
            # Apply reordering
            key_states = torch.gather(key_states, 2, transfer_order[:, None, :, None].expand_as(key_states))
            value_states = torch.gather(value_states, 2, transfer_order[:, None, :, None].expand_as(value_states))

            cache_start_pos = torch.sum(~cache_position, dim=-1)
            if not (cache_start_pos == cache_start_pos[0]).all():
                raise ValueError(
                    f"Heterogeneous batches not supported. "
                    f"All sequences must cache the same number of tokens. "
                    f"Got cache_start_pos={cache_start_pos.tolist()}"
                )
            
            # Store - FIX: Ensure we're updating existing cache entries
            if len(self.key_cache) <= layer_idx:
                # This shouldn't happen in normal flow, but handle it gracefully
                self.key_cache.append(key_states[:, :, cache_start_pos[0]:, :])
                self.value_cache.append(value_states[:, :, cache_start_pos[0]:, :])
            else:
                # Update existing cache entry
                self.key_cache[layer_idx] = key_states[:, :, cache_start_pos[0]:, :]
                self.value_cache[layer_idx] = value_states[:, :, cache_start_pos[0]:, :]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def refresh_cache(self):
        self.key_cache = []
        self.value_cache = []

    def is_empty(self) -> bool:
        return len(self.key_cache) < self.num_hidden_layers or len(self.value_cache) < self.num_hidden_layers

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}")
    
    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __len__(self) -> int:
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length


class HierarchicalDynamicCache(Cache):
    """
    Optimization Level 2: Hierarchical Caching (Stable + Moving).

    Splits the cache into:
    1. Stable Cache: Confirmed tokens (immutable, append-only).
    2. Moving Cache: Recent/Speculative tokens (mutable, frequent reordering).

    The expensive 'gather' reordering is only applied to the small 'Moving' section.
    """
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self._transfer_order = None
        
        # Main storage for confirmed tokens (List of Tensors per layer)
        self.stable_key_cache: List[torch.Tensor] = []
        self.stable_value_cache: List[torch.Tensor] = []
        
        # Temporary storage for speculative/moving tokens
        self.moving_key_cache: List[torch.Tensor] = []
        self.moving_value_cache: List[torch.Tensor] = []

    def get_transfer_cache(self, layer_idx, cache_position, prv_cache_position):
        """Calculate transfer order for moving window only"""
        if layer_idx > 0:
            if self._transfer_order is None:
                return self._compute_transfer_order_for_moving(cache_position, prv_cache_position)
            
            order = self._transfer_order
            if layer_idx == self.num_hidden_layers - 1:
                self._transfer_order = None
            return order
        
        return self._compute_transfer_order_for_moving(cache_position, prv_cache_position)

    def _compute_transfer_order_for_moving(self, moving_cache_pos, moving_prv_pos):
        """
        Computes reordering indices for the MOVING portion only.
        
        The indices returned are LOCAL to the moving window (0, 1, 2, ... moving_len-1).
        """
        B = moving_cache_pos.shape[0]
        device = moving_cache_pos.device
        
        # Current state: new tokens first, then previously cached tokens
        prev_indices = moving_prv_pos.nonzero(as_tuple=True)[1].view(B, -1)
        new_indices = (~moving_prv_pos).nonzero(as_tuple=True)[1].view(B, -1)
        current_order = torch.cat([new_indices, prev_indices], dim=-1)

        # Next state: tokens to drop first, then tokens to keep
        keep_indices = moving_cache_pos.nonzero(as_tuple=True)[1].view(B, -1)
        drop_indices = (~moving_cache_pos).nonzero(as_tuple=True)[1].view(B, -1)
        next_order = torch.cat([drop_indices, keep_indices], dim=-1)

        # Build lookup table
        max_val = moving_cache_pos.shape[1]  # Moving window length
        lookup_table = torch.full((B, max_val), -1, dtype=torch.long, device=device)
        src_indices = torch.arange(current_order.shape[1], device=device).expand(B, -1)
        lookup_table.scatter_(1, current_order, src_indices)

        # Get transfer order
        transfer_order = torch.gather(lookup_table, 1, next_order)
        return transfer_order
    
    def get_confirmed_len_local(self, cache_position_local):
        """Find the contiguous prefix of cached tokens"""
        # Find first False in cache_position
        first_false = (~cache_position_local).nonzero(as_tuple=True)[0]
        
        if first_false.numel() > 0:
            # There's at least one False - confirmed_len is position of first False
            return first_false[0].item()
        else:
            # All True - entire sequence is stable
            return cache_position_local.shape[0]
        
    def get_confirmed_len(self, cache_position):
        """Vectorized confirmed length calculation for batch"""
        if cache_position is None:
            return 0
        # Find first False in each batch element
        B = cache_position.shape[0]
        confirmed_lens = []
        for b in range(B):
            confirmed_lens.append(self.get_confirmed_len_local(cache_position[b]))

        return min(confirmed_lens)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        cache_kwargs = cache_kwargs or {}
        B, n_h, seq_len, h_d = key_states.shape
        prv_cache_position = cache_kwargs.get("prv_cache_position", None)
        cache_position = cache_kwargs.get("cache_position", None)
        
        # 'confirmed_len' tells us where the stable prefix ends.
        # confirmed_len = cache_kwargs.get("confirmed_len", 0)
        # confirmed_len = self.get_confirmed_len(cache_position) # NOTE: repeated work, could be once per model instead of per layer
        confirmed_len_current = self.get_confirmed_len(cache_position)
        confirmed_len_previous = self.get_confirmed_len(prv_cache_position)
        confirmed_len = min(confirmed_len_current, confirmed_len_previous)

        # print(f"Layer {layer_idx} - confirmed_len: {confirmed_len}")
        
        # Validate confirmed_len
        if confirmed_len < 0 or confirmed_len > seq_len:
            raise ValueError(f"confirmed_len must be in [0, {seq_len}], got {confirmed_len}")

        # 1. Initialization Phase (First Step)
        if prv_cache_position is None:
            # Initial fill - store everything as stable for now
            assert len(self.stable_key_cache) <= layer_idx, "Cache should be empty during initialization"

            # Split into stable and moving portions based on confirmed_len
            stable_keys = key_states[:, :, :confirmed_len, :]
            stable_values = value_states[:, :, :confirmed_len, :]
            
            moving_keys = key_states[:, :, confirmed_len:, :]
            moving_values = value_states[:, :, confirmed_len:, :]
            
            # Select only cached tokens from each
            if cache_position is not None:
                # Check: confirmed_len shouldn't exceed number of cached tokens
                num_cached = cache_position.sum(dim=1)
                if (confirmed_len > num_cached).any():
                    raise ValueError(
                        f"confirmed_len ({confirmed_len}) exceeds number of cached tokens. "
                        f"Cached counts per batch: {num_cached.tolist()}"
                    )
                
                stable_mask = cache_position[:, :confirmed_len]
                moving_mask = cache_position[:, confirmed_len:]
                
                if stable_mask.any():
                    # Expand mask to match tensor dimensions and extract True positions
                    expanded_mask = stable_mask[:, None, :, None].expand_as(stable_keys)
                    stable_keys = stable_keys[expanded_mask].view(B, n_h, -1, h_d)
                    
                    expanded_mask = stable_mask[:, None, :, None].expand_as(stable_values)
                    stable_values = stable_values[expanded_mask].view(B, n_h, -1, h_d)
                else:
                    stable_keys = torch.empty(B, n_h, 0, h_d, device=key_states.device, dtype=key_states.dtype)
                    stable_values = torch.empty(B, n_h, 0, h_d, device=value_states.device, dtype=value_states.dtype)
                    
                if moving_mask.any():
                    # Expand mask to match tensor dimensions and extract True positions
                    expanded_mask = moving_mask[:, None, :, None].expand_as(moving_keys)
                    moving_keys = moving_keys[expanded_mask].view(B, n_h, -1, h_d)
                    
                    expanded_mask = moving_mask[:, None, :, None].expand_as(moving_values)
                    moving_values = moving_values[expanded_mask].view(B, n_h, -1, h_d)
                else:
                    moving_keys = torch.empty(B, n_h, 0, h_d, device=key_states.device, dtype=key_states.dtype)
                    moving_values = torch.empty(B, n_h, 0, h_d, device=value_states.device, dtype=value_states.dtype)
            else:
                raise ValueError("cache_position is required for HierarchicalDynamicCache")

            self.stable_key_cache.append(stable_keys)
            self.stable_value_cache.append(stable_values)
            self.moving_key_cache.append(moving_keys)
            self.moving_value_cache.append(moving_values)
            
            # Return combined cache
            if moving_keys.numel() > 0:
                final_keys = torch.cat([stable_keys, moving_keys], dim=2)
                final_values = torch.cat([stable_values, moving_values], dim=2)
            else:
                final_keys = stable_keys
                final_values = stable_values
            
            return final_keys, final_values

        # 2. Hierarchical Reordering Phase
        else:
            input_unmasked_count = (~prv_cache_position).sum(dim=1)[0].item()
            # NOTE: input is given as [input_unmasked_count | input_stable_count | input_moving_count]
            # even though logically it should be [input_stable_count | input_moving_count | input_unmasked_count]
            # and we need [input_stable_count | input_unmasked_count | input_moving_count]
            # TODO: consider more efficient way to do this without concatenation
            # could change convention to be [input_stable_count | input_moving_count | input_unmasked_count]
            # in the actual attention mechanism, but would need to make sure RoPE obeys this order
            key_states = torch.cat([
                key_states[:, :, input_unmasked_count:input_unmasked_count + confirmed_len, :],
                key_states[:, :, :input_unmasked_count, :],
                key_states[:, :, confirmed_len + input_unmasked_count:, :]
            ], dim=2)
            value_states = torch.cat([
                value_states[:, :, input_unmasked_count:input_unmasked_count + confirmed_len, :],
                value_states[:, :, :input_unmasked_count, :],
                value_states[:, :, confirmed_len + input_unmasked_count:, :]
            ], dim=2)

            # Split into stable and moving portions based on confirmed_len
            stable_keys = key_states[:, :, :confirmed_len, :]
            stable_values = value_states[:, :, :confirmed_len, :]
            
            moving_keys = key_states[:, :, confirmed_len:, :]
            moving_values = value_states[:, :, confirmed_len:, :]
            
            # Calculate transfer order only for moving window
            moving_cache_pos = cache_position[:, confirmed_len:]
            moving_prv_pos = prv_cache_position[:, confirmed_len:]
            # don't need for stable portion because it's a prefix of fully cached tokens
            # can verify:
            assert (cache_position[:, :confirmed_len] == True).all(), f"Stable portion of cache_position should be all True but got {cache_position[:, :confirmed_len]}"
            
            if moving_cache_pos.numel() > 0 and moving_prv_pos.numel() > 0:
                local_transfer_order = self.get_transfer_cache(
                    layer_idx, moving_cache_pos, moving_prv_pos
                )
                
                # Apply reordering only to moving part
                moving_keys_reordered = torch.gather(
                    moving_keys, 2, 
                    local_transfer_order[:, None, :, None].expand_as(moving_keys)
                )
                moving_values_reordered = torch.gather(
                    moving_values, 2, 
                    local_transfer_order[:, None, :, None].expand_as(moving_values)
                )
            else:
                moving_keys_reordered = moving_keys
                moving_values_reordered = moving_values

            # Extract only the cached portion of moving window
            cache_start_in_moving = torch.sum(~moving_cache_pos, dim=-1)
            
            # Validate homogeneous batching
            if not (cache_start_in_moving == cache_start_in_moving[0]).all():
                raise ValueError(
                    f"Heterogeneous batches not supported. "
                    f"All sequences must cache the same number of tokens in moving window. "
                    f"Got cache_start_in_moving={cache_start_in_moving.tolist()}"
                )
            
            # Update storage
            new_moving_keys = moving_keys_reordered[:, :, cache_start_in_moving[0]:, :]
            new_moving_values = moving_values_reordered[:, :, cache_start_in_moving[0]:, :]
            if len(self.stable_key_cache) <= layer_idx:
                # First reordering pass for this layer - append
                self.stable_key_cache.append(stable_keys)
                self.stable_value_cache.append(stable_values)
                self.moving_key_cache.append(new_moving_keys)
                self.moving_value_cache.append(new_moving_values)
            else:
                # Update existing entries
                self.stable_key_cache[layer_idx] = stable_keys
                self.stable_value_cache[layer_idx] = stable_values
                self.moving_key_cache[layer_idx] = new_moving_keys
                self.moving_value_cache[layer_idx] = new_moving_values

            # Return combined cache
            if self.moving_key_cache[layer_idx].numel() > 0:
                final_keys = torch.cat(
                    [self.stable_key_cache[layer_idx], self.moving_key_cache[layer_idx]], 
                    dim=2
                )
                final_values = torch.cat(
                    [self.stable_value_cache[layer_idx], self.moving_value_cache[layer_idx]], 
                    dim=2
                )
            else:
                final_keys = self.stable_key_cache[layer_idx]
                final_values = self.stable_value_cache[layer_idx]
            
            return final_keys, final_values

    def refresh_cache(self):
        self.stable_key_cache = []
        self.stable_value_cache = []
        self.moving_key_cache = []
        self.moving_value_cache = []

    def is_empty(self) -> bool:
        return (
            len(self.stable_key_cache) < self.num_hidden_layers or 
            len(self.stable_value_cache) < self.num_hidden_layers or
            len(self.moving_key_cache) < self.num_hidden_layers or
            len(self.moving_value_cache) < self.num_hidden_layers
        )

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.stable_key_cache):
            # Check if moving cache exists and has content
            if layer_idx < len(self.moving_key_cache) and self.moving_key_cache[layer_idx].numel() > 0:
                final_keys = torch.cat(
                    [self.stable_key_cache[layer_idx], self.moving_key_cache[layer_idx]], 
                    dim=2
                )
                final_values = torch.cat(
                    [self.stable_value_cache[layer_idx], self.moving_value_cache[layer_idx]], 
                    dim=2
                )
            else:
                # No moving cache or it's empty, just return stable
                final_keys = self.stable_key_cache[layer_idx]
                final_values = self.stable_value_cache[layer_idx]
            return final_keys, final_values
        else:
            raise KeyError(f"Cache only has {len(self.stable_key_cache)} layers, attempted to access layer with index {layer_idx}")
    
    def __iter__(self):
        for layer_idx in range(len(self.stable_key_cache)):
            yield self.__getitem__(layer_idx)

    def __len__(self) -> int:
        return len(self.stable_key_cache)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.stable_key_cache):
            stable_len = self.stable_key_cache[layer_idx].shape[-2]
            moving_len = self.moving_key_cache[layer_idx].shape[-2] if layer_idx < len(self.moving_key_cache) else 0
            return stable_len + moving_len
        return 0