from pytorch_lightning import Trainer, seed_everything
from project.few_shot import FewShot
from project.dataloader import FewShotDataModule


def test_training_classifier():
    seed_everything(2021)

    model = FewShot()
    dm = FewShotDataModule(ops=model.preprocess)
    dm.prepare_data()
    dm.setup(stage='fit')

    trainer = Trainer(limit_train_batches=30, limit_val_batches=20, max_epochs=15, logger=None)
    trainer.fit(model, dm)

    dm.setup(stage='test')
    results = trainer.test(dm)
    assert results[0]['test_acc'] > 0.8
