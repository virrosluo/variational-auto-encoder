import lightning as L
import torch

from model.config import PredictionOutput

class vae_lightning(L.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        features = self.model.encoder(x)
        mu, log_var = self.model.mu(features), self.model.log_var(features)
        z_sample = self.model.sampling_z_value(mu, log_var)
        kl_loss = self.model.kl_divergence_loss(z_sample, mu, log_var)

        regenerate_img = self.model.decoder(z_sample)
        recon_loss = self.model.reconstruct_loss(regenerate_img, self.model.log_scale, x)

        minus_elbo = (kl_loss - recon_loss).mean()

        self.log_dict({
            'elbo': minus_elbo,
            'kl': kl_loss.mean(),
            'recon_loss': -(recon_loss.mean())
        })
        return minus_elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        features = self.model.encoder(x)
        mu, log_var = self.model.mu(features), self.model.log_var(features)
        z_sample = self.model.sampling_z_value(mu, log_var)
        kl_loss = self.model.kl_divergence_loss(z_sample, mu, log_var)

        regenerate_img = self.model.decoder(z_sample)
        recon_loss = self.model.reconstruct_loss(regenerate_img, self.model.log_scale, x)

        minus_elbo = (kl_loss - recon_loss).mean()

        self.log_dict({
            'elbo': minus_elbo,
            'kl': kl_loss.mean(),
            'recon_loss': -(recon_loss.mean())
        })
        return minus_elbo

    def predict_step(self, batch, batch_idx) -> PredictionOutput:
        x, _ = batch

        features = self.model.encoder(x)
        mu, log_var = self.model.mu(features), self.model.log_var(features)
        z_sample = self.model.sampling_z_value(mu, log_var)
        kl_loss = self.model.kl_divergence_loss(z_sample, mu, log_var)

        regenerate_img = self.model.decoder(z_sample)
        recon_loss = self.model.reconstruct_loss(regenerate_img, self.model.log_scale, x)

        minus_elbo = (kl_loss - recon_loss).mean()

        return PredictionOutput(
            elbo=minus_elbo,
            kl_loss=kl_loss.mean(),
            recon_loss=-(recon_loss.mean()),
            input_image=x,
            reconstruct_image=regenerate_img
        )
        

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        
