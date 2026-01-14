import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG (CPU)
    torch.manual_seed(seed)

    # PyTorch RNG (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (optional but helps reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_next_step_with_start(n, s, idim, odim):
    """
    n: number of sequences
    s: sequence length
    idim: input dimension
    odim: output dimension
    """

    # 1. Generate the actual sequence values (x1 ... x_s)
    seq = torch.randn(n, s, idim)

    # 2. Create a start token x0
    start_token = torch.zeros(n, 1, idim)
    start_token[:, :, 0] = 1.0   # e.g., one-hot start token

    # 3. Build X = [x0, x1, ..., x_{s-1}]
    X = torch.cat([start_token, seq[:, :-1, :]], dim=1)

    # 4. Build y = [x1, x2, ..., x_s]
    y = seq.clone()

    # Convert to numpy for your NumPy RNN
    return X, y


def forward_pass(X, Wx, Wh, Ba, Wy, By):
    #X shape is (s, idim), Wx (idim, h), Wh (h, h)
    #Ba shape is (h), Ht (sxh), Wy (hxo), By is (-xo), A is (sxh)
    s = X.shape[0]
    h = Wx.shape[1]

    A = torch.zeros(s, h)
    Ht = torch.zeros(s, h)

    odim = Wy.shape[1]

    # A is (sxh), however, it is a transition
    # variable we don't keep all s tensors
    # we use A as (-xh)

    y_pred = torch.zeros(s, odim)

    H_prev = torch.zeros(h)
    for i in range(s):
        # X[i] is (-, i), Wx is (i, h)
        # Wh is (h, h), H_prev = (-, h), Ba (_, h)
        # A is (-, h)
        A[i] = X[i] @ Wx + H_prev @ Wh + Ba

        # H[i] is(-xh)
        Ht[i] = torch.tanh(A[i])

        # y_pred is (-, o), Wy is (hxo), H[i] is (-xh), By is (-xo)
        y_pred[i] = Ht[i] @ Wy + By

        #H_prev is (-xh)
        H_prev = Ht[i] 
        
    return Ht, y_pred

def compute_output_gradient(Ht, Wy, By, y):
    #Ht (s, h), Wy (h, o), By is (-, o), y is (s, o)
    #Ht is (s, h)
    s = Ht.shape[0]
    h = Ht.shape[1]
    grad_Ht = torch.rand(s, h)

    for i in range(s):
        # H[i] = (-, h) Wy.T (h, o) by (-, o) y_pred (-, odim)
        y_pred = Ht[i] @Wy + By
        # y_pred (_, o) Wy is (h, o) grad_H[i] = (-, h)
        grad_Ht[i] = 2*(y_pred-y[i]) @ Wy.T


    return grad_Ht
    
def backward_pass(Ht, X, Wh, y_pred, y,  Wy, By):
    # Ht (s, h), grad_Ht (s, h)
    s = X.shape[0]
    h = Wh.shape[1]


    grad_Ht_prev = torch.zeros(h)

    grad_Wx = torch.zeros_like(Wx)
    grad_Wh = torch.zeros_like(Wh)
    grad_Wy = torch.zeros_like(Wy)
    grad_Ba = torch.zeros_like(Ba)
    grad_By = torch.zeros_like(By)


    for i in reversed(range(s)):
        # y_pred[i] is (-, o) and y[i] (_, o) 
        # grad_y = (-, o)
        grad_y = 2 * (y_pred[i] - y[i])

        grad_By = grad_By + grad_y

        # H[i] (-, h), grad_y (-, o)
        # grad_Wy (h, o)
        grad_Wy = grad_Wy + Ht[i].unsqueeze(1) @ grad_y.unsqueeze(0)

        # grad_htotal (-, h)
        grad_htotal = grad_y @ Wy.T + grad_Ht_prev
    
        # grad_A (_, h) grad_htotal (-, h), Ht[i] (-, h)
        grad_A  = grad_htotal *(1 - Ht[i]**2)

        # X[i] (-, i) grad_A = (-, h)
        # grad_Wx (i, h)
        grad_Wx = grad_Wx + X[i].unsqueeze(0).T @ grad_A.unsqueeze(0)

        # grad_A (-, h), Ht[i] = (-, h)
        # grad_Wh (h, h)
       
        h_prev = Ht[i-1] if i > 0 else torch.zeros(h)
        grad_Wh += grad_A.unsqueeze(0).T @ h_prev.unsqueeze(0)

        # grad_A (-, h)
        grad_Ba = grad_Ba + grad_A

        # grad_A (- x h) Wh ((h x h))= (1 x h)
        grad_Ht_prev =  grad_A @ Wh.T
    return grad_Wx, grad_Wh, grad_Ba, grad_Wy, grad_By


def compute_error(y_pred, targets):
    return torch.mean((y_pred - targets)**2)



if __name__ == "__main__":
    odim = 1
    idim = 1
    h = 4 #hidden dimension
    s = 5 #Sequence Length

    n_generated_samples = 200
    n = int(n_generated_samples*0.8)


    X, y = generate_next_step_with_start(n_generated_samples, s, idim, odim)

    X_eval, y_eval = X[n:, :, :],  y[n:, :, :]
    X, y =  X[:n, :, :],  y[:n, :, :]

    Wx = torch.rand(idim, h)
    Wh = torch.rand(h, h)
    Ba = torch.rand(h)

    Wy = torch.rand(h, odim)
    By = torch.rand(odim)

    grad_Wx_total = torch.zeros_like(Wx) 
    grad_Wh_total = torch.zeros_like(Wh) 
    grad_Ba_total = torch.zeros_like(Ba) 
    grad_Wy_total = torch.zeros_like(Wy) 
    grad_By_total = torch.zeros_like(By)

    total_loss = 0.0
    lr = 1e-4

    n_epochs = 1

    for i in range(n_epochs):
        X_i = X[i]
        y_i = y[i]
        #print(Ht.shape)
        Ht, y_pred = forward_pass(X_i, Wx, Wh, Ba, Wy, By)

        loss = compute_error(y_pred, y_i)

        total_loss += loss

        grad_Wx, grad_Wh, grad_Ba, grad_Wy, grad_By = backward_pass(Ht, X_i, Wh, y_pred, y_i,  Wy, By)

        grad_Wx_total += grad_Wx
        grad_Wh_total += grad_Wh
        grad_Ba_total += grad_Ba
        grad_Wy_total += grad_Wy
        grad_By_total += grad_By
    
    Wx -= lr * grad_Wx_total/n
    Wh -= lr * grad_Wh_total/n
    Ba -= lr * grad_Ba_total/n
    Wy -= lr * grad_Wy_total/n
    By -= lr * grad_By_total/n 
    
    print("Batch loss:", total_loss)

    #Eval
    total_eval_error = 0.0
    n_eval = n_generated_samples - n
    for i in range(1):
        X_i = X[i]
        y_i = y[i]
        _, y_pred = forward_pass(X_i, Wx, Wh, Ba, Wy, By)
        error = compute_error(y_pred, y_i)
        total_eval_error += error.item()
    
    print("Evaluation Error: ", total_eval_error / n_eval)

