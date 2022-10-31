import copy


def enumerate_params(params):
    """
    将params里的参数枚举所有可能的组合
    """
    keys = list(params.keys())

    def enumerate_func(index):
        if index == len(keys) - 1:
            res = []
            assert isinstance(params[keys[index]], list)
            for value in params[keys[index]]:
                res.append({keys[index]: value})
            return res
        else:
            last_res = enumerate_func(index + 1)
            res = []
            assert isinstance(params[keys[index]], list)
            for value in params[keys[index]]:
                for last_item in last_res:
                    new_item = copy.deepcopy(last_item)
                    new_item[keys[index]] = value
                    res.append(new_item)
            return res
    return enumerate_func(0)
