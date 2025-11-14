import copy


datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls  #在 datasets 字典中添加一条记录，键是 name，值是这个类 cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args']) # 拷贝一份原始参数，防止修改原数据
        dataset_args.update(args) # 更新拷贝的参数，覆盖或添加新参数
    else:
        dataset_args = dataset_spec['args'] # 没传额外参数，直接用原始参数
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
