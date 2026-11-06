"""Train DeepQN-Net from one LR observation, with optional HR evaluation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
from scipy.io import savemat
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import DeepQNNet
from loss import ZHSRLoss
from utils.data import InternalPairs, load_cube, validate_reference
from utils.training import train_epoch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Observed LR MAT/NPY, already on the intended data scale')
    parser.add_argument('--input-key', default='data')
    parser.add_argument('--input-layout', choices=['CHW', 'HWC'], default='CHW')
    parser.add_argument('--gt', type=Path, help='Optional HR used only for evaluation metrics')
    parser.add_argument('--gt-key', default='gt')
    parser.add_argument('--gt-layout', choices=['CHW', 'HWC'], default='CHW')
    parser.add_argument('--scale', type=int, choices=[2, 4, 8], default=2)
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--stages', type=int, default=3)
    parser.add_argument('--memory-size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=60, help='Rehearsal learning epochs')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lookahead-epochs', type=int, default=0)
    parser.add_argument('--lookahead-lr', type=float, default=0.0005)
    parser.add_argument('--adaptation-epochs', type=int, default=0)
    parser.add_argument('--adaptation-lr', type=float, default=0.0001)
    parser.add_argument('--inner-lr', type=float, default=0.001)
    parser.add_argument('--patch-size', type=int, help='Internal target patch; default min(512/scale, available LR size), rounded to scale')
    parser.add_argument('--pair-workers', type=int, default=8, help='CPU band-convolution threads (not DataLoader processes)')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are never overwritten')
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or not 1 <= args.pair_workers <= 64:
        parser.error('Invalid seed or worker count')
    if min(args.rank, args.stages, args.memory_size) < 1:
        parser.error('rank/stages/memory-size must be positive')
    epochs = [args.epochs, args.lookahead_epochs, args.adaptation_epochs]
    if min(epochs) < 0 or sum(epochs) == 0:
        parser.error('Require nonnegative epoch counts and at least one epoch')
    if any(not np.isfinite(x) or x <= 0 for x in [args.lr, args.lookahead_lr, args.adaptation_lr, args.inner_lr]):
        parser.error('Learning rates must be finite and positive')
    return args


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


def save_result(directory, prefix, model, outputs, config, epoch, stage, metrics):
    """Weights and all four metrics refer to the same evaluated epoch."""
    output = outputs[-1]
    torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                'stage': stage, 'metrics': metrics}, directory/f'{prefix}.pth')
    values = {'sr': output['sr'][0].clamp(0, 1).cpu().numpy(),
              'abundance': output['refined_abundance'][0].cpu().numpy(),
              'endmembers': model.physics.endmembers.weight[:, :, 0, 0].detach().cpu().numpy(),
              'rho': output['rho'][0].cpu().numpy(),
              'learned_kernel': model.physics.degradation.weight.detach().cpu().numpy(),
              'layout': 'CHW', 'epoch': epoch, 'stage': stage}
    values.update(metrics or {})
    savemat(directory/f'{prefix}.mat', values)
    write_json(directory/f'{prefix}.json', {'epoch': epoch, 'stage': stage, 'metrics': metrics})


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    set_seed(args.seed)
    device = torch.device(args.device)
    observation = load_cube(args.input, args.input_key, args.input_layout)
    gt = load_cube(args.gt, args.gt_key, args.gt_layout) if args.gt else None
    if gt is not None:
        validate_reference(observation, gt, args.scale)
        if min(gt.shape[-2:]) < 12:
            raise ValueError('GT dimensions must be >= 12 for the SSIM window')
    model = DeepQNNet(observation.shape[0], args.scale, args.rank, args.stages, args.memory_size).to(device)
    observed_device = observation[None].to(device)
    model.initialize_endmembers(observed_device)
    pairs = InternalPairs(observation, args.scale, args.patch_size, args.seed, args.pair_workers) if args.epochs+args.lookahead_epochs else None
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(bands=observation.shape[0], effective_patch=pairs.patch if pairs else None,
                  architecture='DeepQN-Net', mixing='MLMM with tangent continuation',
                  observation_sha256=file_hash(args.input), gt_sha256=file_hash(args.gt) if args.gt else None,
                  selection='final epoch')
    write_json(args.output/'config.json', config)
    code_root = Path(__file__).resolve().parent
    write_json(args.output/'source_sha256.json', {str(p.relative_to(code_root)): file_hash(p)
               for p in code_root.rglob('*.py') if not any(x in p.parts for x in ('results', '__pycache__'))})
    write_json(args.output/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
               'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': str(device),
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
    np.save(args.output/'observation.npy', observation.numpy(), allow_pickle=False)
    torch.save(model.state_dict(), args.output/'initial.pth')
    criterion = ZHSRLoss()
    epoch = 0
    schedule = [('rehearsal', args.epochs, args.lr), ('lookahead', args.lookahead_epochs, args.lookahead_lr),
                ('adaptation', args.adaptation_epochs, args.adaptation_lr)]
    try:
        with (args.output/'epochs.jsonl').open('w', encoding='utf-8') as log:
            for stage, epochs, learning_rate in schedule:
                if not epochs:
                    continue
                loader = DataLoader(TensorDataset(observation[None]), batch_size=1) if stage == 'adaptation' else DataLoader(pairs, batch_size=1, shuffle=True, num_workers=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
                for _ in range(epochs):
                    started = time.perf_counter()
                    if pairs is not None:
                        pairs.epoch = epoch
                    loss = train_epoch(model, loader, optimizer, criterion, device, stage, args.inner_lr)
                    epoch += 1
                    scheduler.step(loss)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(observed_device)
                        if not torch.isfinite(outputs[-1]['sr']).all():
                            raise FloatingPointError('Non-finite reconstruction')
                        if gt is not None:
                            from utils.metrics import evaluate
                            metrics = evaluate(outputs[-1]['sr'], gt[None].to(device), args.scale)
                        else:
                            metrics = None
                        if epoch == sum(item[1] for item in schedule):
                            if metrics is not None:
                                save_result(args.output, 'best', model, outputs, config, epoch, stage, metrics)
                            save_result(args.output, 'final', model, outputs, config, epoch, stage, metrics)
                    record = dict(epoch=epoch, stage=stage, loss=loss, metrics=metrics,
                                  lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
                    log.write(json.dumps(record)+'\n')
                    log.flush()
                    print(json.dumps(record), flush=True)
    finally:
        if pairs is not None:
            pairs.close()


if __name__ == '__main__':
    main()
