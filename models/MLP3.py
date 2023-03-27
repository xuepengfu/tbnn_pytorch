import torch
import torch.nn as nn


class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(5, 15),
            nn.ReLU(),
            nn.Linear(15, 50),    
            nn.ReLU(),     
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),            
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
 )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')
        
    
    def forward(self, x, y):
        x_lam, x_basis = x, y
        
        C =  self.net(x_lam)
        
        out = (C.view(*C.size(),1) * x_basis).sum(dim=1)

        return out    
    
    
def main():
    blk = MLP3();
    tmp = torch.rand(15,5)
    tmp1 = torch.rand(15,10,9)
    out = blk(tmp,tmp1)

    print(blk)
    print('block:', out.shape)


if __name__ == '__main__':
    main()