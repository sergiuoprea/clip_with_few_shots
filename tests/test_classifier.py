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

    trainer = Trainer(max_epochs=1, logger=logger)
    trainer.fit(model, dm)

    dm.setup(stage='test')
    trainer.test()
