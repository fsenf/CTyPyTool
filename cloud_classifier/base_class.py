
class base_class:
    """
    Provides basic functions of paramter setting 
    """


    def __init__(self, class_variables, **kwargs):

        self.class_variables = class_variables
        self.set_parameters(**kwargs)



    def set_parameters(self, **kwargs):
        # set only valid parameters
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.class_variables)


