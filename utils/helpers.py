import torch


def block_causal_mask(t, n, device):
    mask = torch.triu(torch.ones(t * n, t * n, device=device)).bool()
    for i in range(t):
        mask[i * n : (i + 1) * n, i * n : (i + 1) * n] = True

    return mask


def compute_combined_attn(attn_list, attn_maps):
    if len(attn_list) == 0:
        return attn_maps[0]

    propagated_attns = [attn_maps[0]]

    for prev_attn, attn_map in zip(attn_list, attn_maps[1:]):
        propagated_attns.append(prev_attn @ attn_map)

    combined_attn = torch.stack(propagated_attns, dim=0).mean(dim=0)

    return combined_attn


def pad_batched_slots(slots_dict):

    slot_keys = list(slots_dict.keys())
    slot_values = list(slots_dict.values())

    mask = [torch.ones(slot.shape[:2], device=slot.device) for slot in slot_values]

    max_slots = max(slot_keys)
    padded_slots = []
    padded_mask = []
    for i, slot in enumerate(slot_values):
        padding = max_slots - slot.shape[1]
        padded_slots.append(
            torch.cat(
                [
                    slot,
                    torch.zeros(
                        slot.shape[0],
                        padding,
                        slot.shape[2],
                        device=slot.device,
                    ),
                ],
                dim=1,
            )
        )
        padded_mask.append(
            torch.cat(
                [
                    mask[i],
                    torch.zeros(mask[i].shape[0], padding, device=slot.device),
                ],
                dim=1,
            )
        )

    padded_slots = torch.cat(padded_slots, dim=0)
    padded_mask = torch.cat(padded_mask, dim=0).bool()

    return padded_slots, padded_mask
