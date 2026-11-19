import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        softmax_num = []
        softmax = []
        t = 0
        max_z = np.max(z)
        for i in z:
            s = np.exp(i - max_z)
            t += s
            softmax_num.append(s)
        
        for j in softmax_num:
            softmax.append(j/t)


        return np.round(np.array(softmax), 4)
