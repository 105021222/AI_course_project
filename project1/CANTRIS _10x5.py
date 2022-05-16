import numpy as np
import time
import random
import copy

row,col = 10,5 #board size
class Node():
    def __init__(self):
        self.UCB1=float('inf')
        self.children=[] #record next possible move
        self.pos=-1,-1 #root pos does not matter,so set -1,-1
        self.t=0 #total values
        self.n=0 #total visits
        self.parent=None

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
        if len(self.children)==1:
            return self.children[0]
        r=random.randrange(0,len(self.children))
        return self.children[r]
    
    def update(self,val):
        cur=self
        while cur:
            cur.n+=1
            cur.t+=val
            cur=cur.parent
        cur=self
        while cur.parent:
            cur.UCB1=(cur.t/cur.n)+2*np.sqrt(np.log(cur.parent.n)/cur.n)
            cur=cur.parent

    def add_child(self,i,j):
        child=Node()
        child.pos=i,j
        child.parent=self
        self.children.append(child)

class AI():
    def __init__(self):
     
        self.gameover = False
        self.board = np.zeros([row,col], dtype=int)
        self.stable = True #False:need to clean
        self.step = 0 #total turns , start from 0
        self.turn = -1 
        self.mypoints = 0
        self.oppopoints = 0
        self.board = np.loadtxt("board2.txt", dtype= int)
        self.children=[] #record next possible move
        self.pos=-1,-1 #root pos does not matter,so set -1,-1

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
                    child.children=[]
                    if ((child.step%4)//2) == child.turn:
                        pts = child.make_move(i,j)
                        child.mypoints += pts
                    else:
                        pts = child.make_move(i,j)
                        child.oppopoints += pts
                    child.step += 1
                    child.pos=i,j
                    self.children.append(child)

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
            if ((s.step%4)//2) == s.turn:
                pts = s.make_move(x,y)
                s.mypoints += pts
            else:
                pts = s.make_move(x,y)
                s.oppopoints += pts
            s.step += 1

    def random_sim2(self): #from current state simulate a game to the end,and return the end state's value 
        #the state's value is define as (mypoints-oppopoints)
        s=copy.deepcopy(self)
            
        while True:
            if s.gameover:
                return s.oppopoints-s.mypoints
            choose=[]
            for i in range(row):
                for j in range(col):
                    if s.board[i][j]!=0:
                        choose.append([i,j])
            r=random.randrange(0,len(choose))
            x,y=choose[r]
            if ((s.step%4)//2) == s.turn:
                pts = s.make_move(x,y)
                s.mypoints += pts
            else:
                pts = s.make_move(x,y)
                s.oppopoints += pts
            s.step += 1

    def make_decision(self):# use MCTS to make decision
        if self.step<5:
            s0=copy.deepcopy(self)
            s0.expand_child()
            max_idx=0
            max_jdx=0
            max=float('-inf')
            if s0.step%2==1:
                for i in range(len(s0.children)):
                    if -(s0.children[i].oppopoints-s0.children[i].mypoints)>max:
                        max=-(s0.children[i].oppopoints-s0.children[i].mypoints)
                        max_idx=i
                return s0.children[max_idx].pos
            for i in s0.children:
                i.expand_child()
            for i in range(len(s0.children)):
                for j in range(len(s0.children[i].children)):        
                    if -(s0.children[i].children[j].oppopoints-s0.children[i].children[j].mypoints)>max:
                        max=-(s0.children[i].children[j].oppopoints-s0.children[i].children[j].mypoints)
                        max_jdx=j
                        max_idx=i
            print(max_idx,",",max_jdx,":",-(s0.children[max_idx].children[max_jdx].oppopoints-s0.children[max_idx].children[max_jdx].mypoints))
            return s0.children[max_idx].pos
        time_start=time.time()
        time_end=time.time()
        #let s0=current state ,and run MCTS process for 25 secs
        root=Node()
        count=0
        while(time_end-time_start<25):
            count+=1
            s0=copy.deepcopy(self)
            cur=root
            while not cur.isleaf():
                cur=cur.argmax_UCB1()
                x,y=cur.pos
                if ((s0.step%4)//2) == s0.turn:
                    s0.mypoints+=s0.make_move(x,y)
                else:
                    s0.oppopoints+=s0.make_move(x,y)
                s0.step+=1
            if cur.n!=0:
                if not s0.checkgameover():
                    for i in range(row):
                        for j in range(col):
                            if s0.board[i][j]!=0:
                                cur.add_child(i,j)
                    cur=cur.random_child()
                    x,y=cur.pos
                    if ((s0.step%4)//2) == s0.turn:
                        s0.mypoints+=s0.make_move(x,y)
                    else:
                        s0.oppopoints+=s0.make_move(x,y)
                    s0.step+=1
            val=s0.random_sim()
            #val=s0.random_sim2()
            cur.update(val)
            time_end=time.time()
        print(count)
        max_idx=0
        max=float('-inf')   
        for i in range(len(root.children)):
            if root.children[i].UCB1>max:
                max=root.children[i].UCB1
                max_idx=i
        return root.children[max_idx].pos

    def make_decision2(self):
        s0=copy.deepcopy(self)
        s0.expand_child()
        max_idx=0
        max_jdx=0
        max=float('-inf')
        if s0.step%2==1:
            for i in range(len(s0.children)):
                if (s0.children[i].oppopoints-s0.children[i].mypoints)>max:
                    max=(s0.children[i].oppopoints-s0.children[i].mypoints)
                    max_idx=i
            return s0.children[max_idx].pos
        for i in s0.children:
            i.expand_child()
        for i in range(len(s0.children)):
            for j in range(len(s0.children[i].children)):
                    if (s0.children[i].children[j].oppopoints-s0.children[i].children[j].mypoints)>max:
                        max=(s0.children[i].children[j].oppopoints-s0.children[i].children[j].mypoints)
                        max_jdx=j
                        max_idx=i
        print(max_idx,",",max_jdx,":",s0.children[max_idx].children[max_jdx].oppopoints-s0.children[max_idx].children[max_jdx].mypoints)
        return s0.children[max_idx].pos
        

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
        self.turn=1

        #start playing    
        while not self.gameover:
            print('Turn:', self.step)
            if ((self.step%4)//2) == self.turn:
                print('It\'s your turn')
                x,y = self.make_decision()
                print(f"Your move is {x},{y}.")
                # [x,y] = [int(x) for x in input("Enter the move : ").split()]
                assert (0<=x and x<=row-1 and 0<=y and y<=col-1)
                assert (self.board[x][y]>0)
                pts = self.make_move(x,y)
                self.mypoints += pts
                print(f'You get {pts} points')  
                self.show_board()

            else:
                print('It\'s opponent\'s turn')
                #x,y = self.rand_select() # can use this while testing ,close it when you submit
                #[x,y] = [int(x) for x in input("Enter the move : ").split()] #open it when you submit
                x,y = self.make_decision2()
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
    '''
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
    print("lose:",lose)
