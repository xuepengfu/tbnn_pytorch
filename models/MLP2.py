import torch
import torch.nn as nn


class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(5, 15),
            nn.ELU(),
            nn.Linear(15, 50),    
            nn.ELU(),     
            nn.Linear(50, 50),
            nn.ELU(),
            nn.Linear(50, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),            
            nn.Linear(150, 50),
            nn.ELU(),
            nn.Linear(50, 50),
            nn.ELU(),
            nn.Linear(50, 15),
            nn.ELU(),
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
    blk = MLP2();
    tmp = torch.rand(15,5)
    tmp1 = torch.rand(15,10,9)
    out = blk(tmp,tmp1)

    print(blk)
    print('block:', out.shape)


if __name__ == '__main__':
    main()