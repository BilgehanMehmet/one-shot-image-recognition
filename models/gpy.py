import GPy
import GPyOpt
from model import get_val_acc,create_and_train

current_model_number = 0


def main():


    hyperparameters = [{'name': 'lr1', 'type': 'continuous',
                        'domain': (10e-4, 1)},
                       {'name': 'lr2', 'type': 'continuous',
                        'domain': (10e-4, 10e-1)},
                       {'name': 'lr3', 'type': 'continuous',
                        'domain': (10e-4, 10e-1)},
                       {'name': 'lr4', 'type': 'continuous',
                        'domain': (10e-4, 10e-1)},
                       {'name': 'lr5', 'type': 'continuous',
                        'domain': (10e-4, 10e-1)},
                       {'name': 'lr6', 'type': 'continuous',
                        'domain': (10e-4, 10e-1)},
                       {'name': 'm1', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'm2', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'm3', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'm4', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'm5', 'type': 'continuous',
                        'domain': (0.0, 1.0)},
                       {'name': 'm6', 'type': 'continuous',
                        'domain': (0.0, 1.0)}
                    ]


    def bayesian_optimization_function(x):
        learning_rate1 = float(x[:, 0])
        learning_rate2 = float(x[:, 1])
        learning_rate3 = float(x[:, 2])
        learning_rate4 = float(x[:, 3])
        learning_rate5 = float(x[:, 4])
        learning_rate6 = float(x[:, 5])
        momentum1 = float(x[:, 6])
        momentum2 = float(x[:, 7])
        momentum3 = float(x[:, 8])
        momentum4 = float(x[:, 9])
        momentum5 = float(x[:, 10])
        momentum6 = float(x[:, 11])
        
        hyperparameters = [
                            learning_rate1,
                            learning_rate2,
                            learning_rate3,
                            learning_rate4,
                            learning_rate5,
                            learning_rate6,
                            momentum1,
                            momentum2,
                            momentum3,
                            momentum4,
                            momentum5,
                            momentum6
                  ]
        acc = create_and_train(hyperparameters,train_data,train_data,test_data,epochs=2,batch_size=256,drawer_size=2)
        tf.reset_default_graph()
        return 1 - acc

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=bayesian_optimization_function, domain=hyperparameters)

    optimizer.run_optimization(max_iter=15)

    print("optimized parameters: {0}".format(optimizer.x_opt))
    print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))


if __name__ == "__main__":
    main()