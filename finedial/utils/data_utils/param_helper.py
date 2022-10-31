class ParamDict(dict):
    def __init__(self, params):
        dict.__init__(self)
        for key in params.keys():
            value = params[key]
            if isinstance(value, dict):
                self[key] = ParamDict(params=value)
            elif isinstance(value, list):
                def check(sub_param):
                    if isinstance(sub_param, dict):
                        sub_param = ParamDict(params=sub_value)
                    elif isinstance(sub_param, list):
                        for sub_idx, sub_sub_value in enumerate(sub_param):
                            sub_param[sub_idx] = check(sub_sub_value)
                    else:
                        pass
                    return sub_param
                for idx, sub_value in enumerate(value):
                    value[idx] = check(sub_value)
                self[key] = params[key]
            else:
                self[key] = params[key]


        self.__dict__ = self