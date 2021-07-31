from pytorch_lightning import Trainer, seed_everything
from project.few_shot import FewShot
from project.dataloader import FewShotDataModule
from pytorch_lightning.loggers.neptune import NeptuneLogger


def test_classifier():
    seed_everything(2021)

    model = FewShot()
    dm = FewShotDataModule(ops=model.preprocess)
    dm.prepare_data()
    dm.setup(stage='fit')

    logger = NeptuneLogger(offline_mode=True)

    trainer = Trainer(limit_train_batches=30, limit_val_batches=20, max_epochs=1, logger=logger)
    trainer.fit(model, dm)

    dm.setup(stage='test')
    trainer.test()
    assert trainer.checkpoint_callback.best_model_score.item() > 0.7


if __name__=='__main__':
    test_classifier()