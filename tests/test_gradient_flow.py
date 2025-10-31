import torch
from models import HPCNet

def test_gradient_flow():
    model = HPCNet(["class_0"])
    model.add_compositional_prompt()
    model.add_instance_prompt("class_1")
    
    images = [torch.rand(3, 224, 224)]
    logits, _, _, _ = model(images, ["class_0", "class_1"])
    loss = logits.sum()
    loss.backward()

    assert model.prompt_bank.fp_prompts.grad is not None
    assert model.prompt_bank.fp_prompts.grad.abs().sum() > 0
    assert model.prompt_bank.cp_alphas[0].grad is not None
    print("✅ Gradient flow through hierarchy verified.")

if __name__ == "__main__":
    test_gradient_flow()
