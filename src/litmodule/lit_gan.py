import lightning as L
import torch
import torchmetrics.image
from torch import nn, optim



class LitMNISTGAN(L.LightningModule):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, num_classes: int, lr: float = 0.0, beta_1: float = 0.0, beta_2: float = 0.99):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.criterion = nn.CrossEntropyLoss()
        self.fid_score = torchmetrics.image.fid.FrechetInceptionDistance(
            
        )
        self.is_score = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def update_discriminator(x, class_ids, discriminator, generator, optimizer, params):
        bs = x.size(0)
        device = x.device

        optimizer.zero_grad()

        # for data (ground-truth) distribution
        disc_real = discriminator(x, class_ids, flg_train=True)
        loss_real = eval('compute_loss_'+args.model)(disc_real, loss_type='real')

        # for generator distribution
        latent = torch.randn(bs, params["dim_latent"], device=device)
        img_fake = generator(latent, class_ids)
        disc_fake = discriminator(img_fake.detach(), class_ids, flg_train=True)
        loss_fake = eval('compute_loss_'+args.model)(disc_fake, loss_type='fake')


        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer.step()

    def update_generator(num_class, discriminator, generator, optimizer, params, device):
        optimizer.zero_grad()

        bs = params['batch_size']
        latent = torch.randn(bs, params["dim_latent"], device=device)

        class_ids = torch.randint(num_class, size=(bs,), device=device)
        batch_fake = generator(latent, class_ids)

        disc_gen = discriminator(batch_fake, class_ids, flg_train=False)
        loss_g = - disc_gen.mean()
        loss_g.backward()
        optimizer.step()

    def compute_loss_gan(disc, loss_type):
        assert (loss_type in ['real', 'fake'])
        if 'real' == loss_type:
            loss = (1. - disc).relu().mean() # Hinge loss
        else: # 'fake' == loss_type
            loss = (1. + disc).relu().mean() # Hinge loss

        return loss

    def compute_loss_san(disc, loss_type):
        assert (loss_type in ['real', 'fake'])
        if 'real' == loss_type:
            loss_fun = (1. - disc['fun']).relu().mean() # Hinge loss for function h
            loss_dir = - disc['dir'].mean() # Wasserstein loss for omega
        else: # 'fake' == loss_type
            loss_fun = (1. + disc['fun']).relu().mean() # Hinge loss for function h
            loss_dir = disc['dir'].mean() # Wasserstein loss for omega
        loss = loss_fun + loss_dir

        return loss

    def step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, target = batch
        pred = self.model(x)
        loss = self.criterion(pred, target)
        return loss, pred, target

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy(pred, target)
        self.log(
            "train/accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy(pred, target)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, target = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_accuracy(pred, target)
        self.log(
            "test/accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        return [optimizer_G, optimizer_D], []

