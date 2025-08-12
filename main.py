import argparse
from model import UNet
import utils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataset import NumDataset
from sampler import Sampler
import torch.nn.functional as F
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", 
                        action="store_true")
    parser.add_argument("--save_model",
                        action="store_true")
    parser.add_argument("--model_load_dir",
                        type=str,
                        default="models/model.pth")
    parser.add_argument("--model_save_dir",
                        type=str,
                        default="models/model.pth")
    parser.add_argument("--n_samples",
                        type=int,
                        default=5)
    parser.add_argument("--sample_save_dir",
                        type=str,
                        default="./samples")
    parser.add_argument("--model_config",
                        type=str,
                        default="./configs/model.yaml")

    parser = parser.parse_args()

    config = OmegaConf.load(f"{parser.model_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(enc_chs=config.model.enc_chs,
                 dec_chs=config.model.dec_chs, 
                 num_class=config.model.num_class, 
                 retain_dim=config.model.retain_dim, 
                 out_size=config.model.out_size, 
                 time_emb_dim=config.model.time_emb_dim)
    
    sampler = Sampler(model,
                    config.sampler.timesteps, 
                    config.sampler.start, 
                    config.sampler.end, 
                    config.sampler.img_size)
    
    if parser.train:
        data = NumDataset()
        dataloader = DataLoader(data, batch_size=config.training.batch_size, shuffle=True)

        lr = config.training.lr
        opt = torch.optim.Adam(model.parameters(), lr)
        epochs = config.training.epochs

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                t = torch.randint(0, sampler.T, size=(len(batch[0]), ), device=device)
                x_noisy, noise = sampler.forward_diffusion(batch[0], t, device)
                noise_pred = model(x_noisy, t)
                loss = F.l1_loss(noise, noise_pred)

                opt.zero_grad()
                loss.backward()
                opt.step()
                
            print(f"Epoch {epoch} Loss: {loss.item()} ")

        if parser.save_model:
            utils.save_model(model, parser.model_save_dir)
    
    else: 
        model = utils.load_model(model, parser.model_load_dir)
        sampler = Sampler(model,
                    config.sampler.timesteps, 
                    config.sampler.start, 
                    config.sampler.end, 
                    config.sampler.img_size)

    for n in range(parser.n_samples):
        sample = sampler.sample_image(device)

        utils.save_sample(sample, parser.sample_save_dir)

if __name__ == "__main__":
    main()