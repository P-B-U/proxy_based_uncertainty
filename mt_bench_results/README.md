File Structure

1. sharegpt2dolly_7b.jsonl - Llama 7b
   - sharegpt_sft: SFT base model
   - sharegpt2dolly_random: random curriculum
   - sharegpt2dolly_espistemic_up: sorted by increasing epistemic uncertainty
   - sharegpt2dolly_espistemic_down: sorted by decreasing epistemic uncertainty
   - sharegpt2dolly_aleatoric_up: sorted by increasing aleatoric uncertainty
   - sharegpt2dolly_aleatoric_down: sorted by decreasing aleatoric uncertainty
   - sharegpt2dolly_balanced_entropy_up: sorted by increasing balanced_entropy
   - sharegpt2dolly_balanced_entropy_down: sorted by decreasing balanced_entropy
   - sharegpt2dolly_random_udpo: random + udpo
   - sharegpt2dolly_aleatoric_up_udpo: sorted by increasing aleatoric uncertainty + udpo
  
2. sharegpt2dolly_13b.jsonl - Llama 13b
   - sharegpt_sft_13b: SFT base model
   - sharegpt2dolly_random_13b: random curriculum
   - sharegpt2dolly_espistemic_up_13b: sorted by increasing epistemic uncertainty
   - sharegpt2dolly_espistemic_down_13b: sorted by decreasing epistemic uncertainty
   - sharegpt2dolly_aleatoric_up_13b: sorted by increasing aleatoric uncertainty
   - sharegpt2dolly_aleatoric_down_13b: sorted by decreasing aleatoric uncertainty
   - sharegpt2dolly_balanced_entropy_up_13b: sorted by increasing balanced_entropy
   - sharegpt2dolly_balanced_entropy_down_13b: sorted by decreasing balanced_entropy
   - sharegpt2dolly_random_udpo_13b: random + udpo
   - sharegpt2dolly_aleatoric_up_udpo_13b: sorted by increasing aleatoric uncertainty + udpo
