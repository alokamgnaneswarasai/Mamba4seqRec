import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from mamba4rec import Mamba4Rec,SASRec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


if __name__ == '__main__':

    # config = Config(model=Mamba4Rec, config_file_list=['config.yaml'])
    config = Config(model=SASRec, config_file_list=['config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    print(type(dataset))
    # dataset.data_augmentation()
    
    # methods = [method for method in dir(dataset) if callable(getattr(dataset, method))]
    # print(methods)
    # dataset.prepare_data_augmentation()
    # uid_list = dataset.uid_list
    # iid_list = dataset.iid_list
    # logger.info(f"uid_list: {uid_list}")
    # logger.info(f"iid_list: {iid_list}")
    

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    print(type(train_data))
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    # model = Mamba4Rec(config, train_data.dataset).to(config['device'])
    model = SASRec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

