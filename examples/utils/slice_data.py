import numpy as np

def sliceDataAlongAxis(data, fractions, axis):
    # Assume a datase with first dimensiton denotiong its principal dimension
    data_size = data.shape[axis]
    fractions_ = np.zeros_like(fractions, dtype=int)

    total_size = 0
    for i, fraction in enumerate(fractions):
        total_size += int(data_size * fraction)
    remain = data_size - total_size

    remain_added = False
    slices = ()
    for i, fraction in enumerate(fractions):
        fractions_[i] = int(data_size * fraction) 
        if fractions_[i] != 0 and not remain_added:
            fractions_[i] += remain
            remain_added = True
        if i > 0:
            fractions_[i] += fractions_[i-1]
            slice = data.take(range(fractions_[i-1], fractions_[i]), axis)    
        else:
            slice = data.take(range(0, fractions_[i]), axis)    
        slices += (slice,)
        
    return slices