if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero import MLP


model = MLP((10, 1))
print(model)
