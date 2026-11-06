"""Rehearsal and first-order lookahead learning, with optional final adaptation."""
import torch


def train_epoch(model, loader, optimizer, criterion, device, stage, inner_lr=1e-3):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        observation = batch[0].to(device)
        target = batch[1].to(device) if stage != 'adaptation' else None
        optimizer.zero_grad(set_to_none=True)
        if stage == 'lookahead':
            # First-order approximation used in the original implementation.
            original = {name: value.detach().clone() for name, value in model.state_dict().items()}
            try:
                inner = criterion(model, model(observation), observation, observation_consistency=True)
                if not torch.isfinite(inner):
                    raise FloatingPointError('Non-finite lookahead inner loss')
                inner.backward()
                # The temporary SGD step needs the same clipping as outer updates.
                # Without it, a large inner gradient can corrupt all proxy weights.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.add_(parameter.grad, alpha=-inner_lr)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model, model(observation), observation, target, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite lookahead outer loss')
                loss.backward()
            finally:
                # Keep outer gradients at proxy weights; restore original parameters.
                model.load_state_dict(original)
        else:
            loss = criterion(model, model(observation), observation, target, stage == 'adaptation')
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item()
        count += 1
    if not count:
        raise ValueError('Empty training loader')
    return total / count
