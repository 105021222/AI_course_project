import numpy as np
import random
import copy
import time
row,col = 6,3 #board size


class AI():
    def __init__(self):
     
        self.gameover = False
        self.board = np.zeros([row,col], dtype=int)
        self.stable = True #False:need to clean
        self.step = 0 #total turns , start from 0
        self.turn = -1 
        self.mypoints = 0
        self.oppopoints = 0
        self.board = np.loadtxt("board.txt", dtype= int)
        self.children=[] #record next possible move
        self.UCB1=float('inf') 
        self.pos=-1,-1 #root pos does not matter,so set -1,-1
        self.t=0 #total values
        self.n=0 #total visits
        self.parent=None
        
        # #make board
        tmp = (np.arange(row)/2)+1
        while(1):
            for i in range(col):
                np.random.shuffle(tmp)
                self.board[:,i] = tmp
            if self.checkstable() == True:
                break
        #self.show_board()
        
    def checkstable(self):
        for r in range(len(self.board)):
            for c in range(len(self.board[0])-2):
                if abs(self.board[r][c]) == abs(self.board[r][c+1]) == abs(self.board[r][c+2]) != 0:
                    self.stable = False
                    return self.stable
        self.stable = True
        return self.stable

    def drop(self): # drop the tile 
        for c in range(col):
            if self.board[:,c].sum()!=0:
                k = -len(self.board[:,c][self.board[:,c]>0])
                self.board[k:,c] = self.board[:,c][self.board[:,c]>0]
                self.board[:k,c] = 0

    def checkgameover(self): #any of last row == 0
        self.gameover = np.any(self.board[-1] == 0)
        return self.gameover

    def clean(self): #clean the tile , return points
        unstable = 1
        points = 0
        while(unstable):
            unstable = 0
            for i in range(row):
                for j in range(col-2):
                    if abs(self.board[i][j]) == abs(self.board[i][j+1]) == abs(self.board[i][j+2]) != 0:
                        self.board[i][j] = self.board[i][j+1] = self.board[i][j+2] = -abs(self.board[i][j])
                        unstable = 1
            points -= self.board[self.board<0].sum()
            self.board[self.board<0] = 0
            self.drop()
        self.checkgameover()        
        return points

    def expand_child(self): #expand node,add child to tree
        if self.checkgameover():
            return
        for i in range(row):
            for j in range(col):
                if self.board[i][j]!=0: #(i,j) is legal move so make child node and add to self.children
                    child=copy.deepcopy(self)
                    child.n=0
                    child.t=0
                    child.children=[]
                    child.UCB1=float('inf')
                    child.parent=self
                    if (child.step%2) == child.turn:
                        pts = child.make_move(i,j)
                        child.mypoints += pts
                    else:
                        pts = child.make_move(i,j)
                        child.oppopoints += pts
                    child.step += 1
                    child.pos=i,j
                    self.children.append(child)

    def isleaf(self):
        return self.children==[]

    def argmax_UCB1(self): #pick child which has max. UCB1
        max=float('-inf')
        choose=[]
        for i in self.children:
            if i.UCB1>max:
                max=i.UCB1
                choose=[]
                choose.append(i)
            elif i.UCB1==max:
                choose.append(i)
        if len(choose)==1:
            return choose[0]
        else:
            #if have multiple choice,then pick one at random
            x=random.randrange(0,len(choose))
            return choose[x]
    
    def random_child(self): #pick child at random,and if it has no child,return itself
        if len(self.children)==0:
            return self
        if len(self.children)==1:
            return self.children[0]
        r=random.randrange(0,len(self.children))
        return self.children[r]

    def random_sim(self): #from current state simulate a game to the end,and return the end state's value 
        #the state's value is define as (mypoints-oppopoints)
        s=copy.deepcopy(self)
        while True:
            if s.gameover:
                return s.mypoints-s.oppopoints
            choose=[]
            for i in range(row):
                for j in range(col):
                    if s.board[i][j]!=0:
                        choose.append([i,j])
            r=random.randrange(0,len(choose))
            x,y=choose[r]
            if (s.step%2) == s.turn:
                pts = s.make_move(x,y)
                s.mypoints += pts
            else:
                pts = s.make_move(x,y)
                s.oppopoints += pts
            s.step += 1

    def update(self,val): #update n,t,UCB1 value in tree
        cur=self
        while cur:
            cur.n+=1
            cur.t+=val
            cur=cur.parent
        cur=self
        while cur.parent:
            cur.UCB1=(cur.t/cur.n)+2*np.sqrt(np.log(cur.parent.n)/cur.n)
            cur=cur.parent
    
    def make_decision_2(self):
        s0=copy.deepcopy(self)
        s0.expand_child()
        max_idx=0
        max_jdx=0
        max=float('-inf')
        
        for i in range(len(s0.children)):
            if (s0.children[i].oppopoints-s0.children[i].mypoints)>max:
                max=(s0.children[i].oppopoints-s0.children[i].mypoints)
                max_idx=i
        return s0.children[max_idx].pos     

    def make_decision(self): # use MCTS to make decision
        time_start=time.time()
        time_end=time.time()
        #let s0=current state ,and run MCTS process for 25 secs
        s0=copy.deepcopy(self)
        while(time_end-time_start<10):
            cur=s0
            while not cur.isleaf():
                cur=cur.argmax_UCB1()
                
            if cur.n!=0:
                cur.expand_child()
                cur=cur.random_child()
                
            val=cur.random_sim()
            cur.update(val)
            time_end=time.time()
        '''for i in s0.children:
            print(i.UCB1)
            print(i.checkgameover())
            print(i.isleaf())'''
        #if we can pick the move that can end the game and also win the game,pick it
        if len(s0.children)<=(row*(col-1)+1):
            '''print('check')'''
            for i in s0.children:
                if i.isleaf() and i.checkgameover():
                    if (i.mypoints-i.oppopoints)>0:
                        return i.pos
        #pick the move that has max. UCB1
        max_idx=0
        max=float('-inf')   
        for i in range(len(s0.children)):
            if s0.children[i].n>max:
                max=s0.children[i].n
                max_idx=i
        return s0.children[max_idx].pos
               
        # return format : [x,y]
        # Use AI to make decision !
        # random is only for testing !

    
    def rand_select(self):
        p = 0
        while not(p):
            x= np.random.randint(row)
            y= np.random.randint(col)
            p = self.board[x][y]
        return [x,y]

    def make_move(self, x, y):
        pts = self.board[x][y]
        self.board[1:x+1,y] = self.board[0:x,y]
        self.board[0][y] = 0

        if self.checkgameover(): 
            return pts
        
        pts += self.clean()
        return pts
    
        
    def start(self):
        print("Game start!")      
        print('――――――――――――――――――')
        self.show_board()
        #self.turn = int(input("Set the player's order(0:first, 1:second): "))
        self.turn=0

        #start playing    
        while not self.gameover:
            print('Turn:', self.step)
            if (self.step%2) == self.turn:
                print('It\'s your turn')
                x,y = self.make_decision()
                print(f"Your move is {x},{y}.")
                #[x,y] = [int(x) for x in input("Enter the move : ").split()]#
                assert (0<=x and x<=row-1 and 0<=y and y<=col-1)
                assert (self.board[x][y]>0)
                pts = self.make_move(x,y)
                self.mypoints += pts
                print(f'You get {pts} points')  
                self.show_board()

            else:
                print('It\'s opponent\'s turn')
                #x,y = self.rand_select() # can use this while testing ,close it when you submit
                [x,y] = [int(x) for x in input("Enter the move : ").split()] #open it when you submit
                #x,y = self.make_decision_2()
                assert (0<=x and x<=row-1 and 0<=y and y<=col-1)
                assert (self.board[x][y]>0)
                print(f"Your opponent move is {x},{y}.")
                pts = self.make_move(x,y)
                self.oppopoints += pts
                print(f'Your opponent\'s get {pts} points')
                self.show_board()

            self.step += 1

        #gameover
        if self.mypoints > self.oppopoints:
            print('You win!')
            return 1
        elif self.mypoints < self.oppopoints:
            print('You lose!')
            return -1
        else:
            print('Tie!')
            return 0

    def show_board(self):
        print('my points:', self.mypoints)
        print('opponent\'s points:', self.oppopoints)
        print('The board is :')
        print(self.board)
        print('――――――――――――――――――')

if __name__ == '__main__':
 
    game = AI()
    game.start()
    '''
    win=0
    lose=0
    for i in range(10):
        game = AI()
        x=game.start()
        if x==1:
            win+=1
        elif x==-1:
            lose+=1
    print("win:",win)
    print("lose:",lose)'''