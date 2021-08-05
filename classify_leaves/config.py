train_name = 'new_accelerator'

default_config = dict(
    batch_size=64,
    num_epoch=5,
    learning_rate=3e-4,             # learning rate of Adam
    weight_decay=0.001,             # weight decay 

    train_name = 'new_accelerator',
    warm_up_epochs=10,
    model_path='./model/'+train_name+'_model.ckpt',
    saveFileName='./result/'+train_name+'_pred.csv',
    num_workers=2,
    model_name='effnetv2',

)

