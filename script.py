import numpy as np
import os

path = "exp/ngc_transformer/component/custom"


def load_npz(file_name):
    file_path = os.path.join(path, file_name)

    if not os.path.exists(file_path):
        print(f"\n[NOT FOUND] {file_name}")
        return

    print("\n" + "=" * 120)
    print(f"FILE: {file_name}")
    print("=" * 120)

    data = np.load(file_path)

    for key in data.files:
        print(f"\nKEY: {key}")
        print("-" * 80)
        print(data[key])
        print("-" * 80)


print("\n\n##############################")
print("EMBEDDING FLOW")
print("##############################")


print("\n[1] self.embedding.z_embed.zF")
load_npz("z_embed.npz")

print("\n[2] self.embedding.W_embed.inputs")
print("Connected from z_embed.zF")

print("\n[3] self.embedding.W_embed.outputs")
load_npz("W_embed.npz")

print("\n[4] self.reshape_3d_to_2d_embed.inputs")
print("Connected from W_embed.outputs")

print("\n[5] self.reshape_3d_to_2d_embed.outputs")
load_npz("reshape_3d_to_2d_embed.npz")

print("\n[6] self.blocks[0].attention.z_qkv.j")
print("Connected from reshape_3d_to_2d_embed.outputs")

print("\n[7] self.embedding.e_embed.mu")
load_npz("e_embed.npz")

print("\n[8] self.blocks[0].attention.z_qkv.z")
load_npz("block0_z_qkv.npz")

print("\n[9] self.embedding.e_embed.target")
print("Connected from block0_z_qkv.z")


# =========================
# BLOCKS
# =========================

n_layers = 2   # change if needed

for blocks in range(n_layers):

    print("\n\n")
    print("#" * 120)
    print(f"BLOCK {blocks}")
    print("#" * 120)

    # -------------------------
    # z_qkv -> W_q/W_k/W_v
    # -------------------------

    print(f"\n[BLOCK {blocks}] z_qkv.zF")
    load_npz(f"block{blocks}_z_qkv.npz")

    print(f"\n[BLOCK {blocks}] W_q")
    load_npz(f"block{blocks}_W_q.npz")

    print(f"\n[BLOCK {blocks}] W_k")
    load_npz(f"block{blocks}_W_k.npz")

    print(f"\n[BLOCK {blocks}] W_v")
    load_npz(f"block{blocks}_W_v.npz")

    # -------------------------
    # reshape qkv
    # -------------------------

    print(f"\n[BLOCK {blocks}] reshape_2d_to_3d_q")
    load_npz(f"block{blocks}_reshape_2d_to_3d_q.npz")

    print(f"\n[BLOCK {blocks}] reshape_2d_to_3d_k")
    load_npz(f"block{blocks}_reshape_2d_to_3d_k.npz")

    print(f"\n[BLOCK {blocks}] reshape_2d_to_3d_v")
    load_npz(f"block{blocks}_reshape_2d_to_3d_v.npz")

    # -------------------------
    # attention block
    # -------------------------

    print(f"\n[BLOCK {blocks}] attention block")
    load_npz(f"block{blocks}_attn_block.npz")

    print(f"\n[BLOCK {blocks}] reshape_3d_to_2d")
    load_npz(f"block{blocks}_reshape_3d_to_2d.npz")

    # -------------------------
    # e_qkv
    # -------------------------

    print(f"\n[BLOCK {blocks}] e_qkv")
    load_npz(f"block{blocks}_e_qkv.npz")

    # -------------------------
    # z_attn
    # -------------------------

    print(f"\n[BLOCK {blocks}] z_attn")
    load_npz(f"block{blocks}_z_attn.npz")

    # -------------------------
    # W_attn_out
    # -------------------------

    print(f"\n[BLOCK {blocks}] W_attn_out")
    load_npz(f"block{blocks}_W_attn_out.npz")

    print(f"\n[BLOCK {blocks}] e_attn")
    load_npz(f"block{blocks}_e_attn.npz")

    # -------------------------
    # MLP
    # -------------------------

    print(f"\n[BLOCK {blocks}] z_mlp")
    load_npz(f"block{blocks}_z_mlp.npz")

    print(f"\n[BLOCK {blocks}] W_mlp1")
    load_npz(f"block{blocks}_W_mlp1.npz")

    print(f"\n[BLOCK {blocks}] e_mlp1")
    load_npz(f"block{blocks}_e_mlp1.npz")

    print(f"\n[BLOCK {blocks}] z_mlp2")
    load_npz(f"block{blocks}_z_mlp2.npz")

    print(f"\n[BLOCK {blocks}] W_mlp2")
    load_npz(f"block{blocks}_W_mlp2.npz")

    print(f"\n[BLOCK {blocks}] e_mlp")
    load_npz(f"block{blocks}_e_mlp.npz")


# =========================
# OUTPUT
# =========================

print("\n\n##############################")
print("OUTPUT FLOW")
print("##############################")

print("\n[OUTPUT] z_out")
load_npz("z_out.npz")

print("\n[OUTPUT] W_out")
load_npz("W_out.npz")

print("\n[OUTPUT] z_actfx")
load_npz("z_actfx.npz")

print("\n[OUTPUT] e_out")
load_npz("e_out.npz")

print("\n[OUTPUT] z_target")
load_npz("z_target.npz")
