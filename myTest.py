import functools
# from functools import wraps
def ZSQ(func):
    @functools.wraps(func)
    def inner(*args,**kwargs):
        print("_"*30)
        func(*args,**kwargs)
    return inner
@ZSQ
def myfunc(*argc,**kwargs):
    print(argc,kwargs)

myfunc(1,2,3,c=4)

print(myfunc.__name__)