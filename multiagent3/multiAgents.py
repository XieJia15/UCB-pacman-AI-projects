# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
"""Q1:反馈搜索"""
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # 获取当前游戏状态 其中G表示为Ghost %表示为墙 角标表示pacman 角标方向代表上一次选择的方向
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print 'successorGameState\n',successorGameState

        # 获取这样移动后新的位置
        newPos = successorGameState.getPacmanPosition()
        # print 'newPos',newPos

        # 获取食物在图中的分布（二维数组，有失误为T没食物为F)
        newFood = successorGameState.getFood()
        curFood = currentGameState.getFood()
        # print 'newFood',newFood

        # 获取Ghost的位置
        newGhostStates = successorGameState.getGhostStates()
        # print 'ghostState',newGhostStates[0].getPosition()

        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # 对这个选择评估的分数
        currscore = 0

        if action == "Stop":
            return -100

        # 如果当前状态能够使ghost害怕，将所有的时间加入进来
        for st in newScaredTimes:
            currscore += st

        # 根据Ghost所在的位置，获取与当前位置的距离
        ghost_distances = []
        for gs in newGhostStates:
            ghost_distances += [manhattanDistance(gs.getPosition(), newPos)]

        # 获取food所在的所有pos
        foodList = newFood.asList()
        curfoodList = curFood.asList()

        # 获取food所在的所有wall
        wallList = currentGameState.getWalls().asList()

        # 保存food的距离
        food_distences = []

        # 获取所有食物到达当前位置的距离
        for foodpos in foodList:
            food_distences += [manhattanDistance(newPos, foodpos)]

        # 对食物的距离取反
        inverse_food_distences = 0;
        if len(food_distences) > 0 and min(food_distences) > 0:
            inverse_food_distences = 1.0 / min(food_distences)
        # 考虑了ghost与当前的距离，其权值更大
        currscore += min(ghost_distances) * (inverse_food_distences ** 4)
        # 获取当前系统判定的分数 又可能当前吃到了豆子 分数更高些
        currscore += successorGameState.getScore()
        if newPos in curfoodList:
            currscore = currscore * 1.1
        return currscore
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

"""Q2:minmax搜索"""
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        n = gameState.getNumAgents()
        depth_ = self.depth

        def max_value(state, currentDepth):
            # 当前深度加一
            currentDepth = currentDepth + 1
            # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            # 初始化v
            v = float('-Inf')
            # 对每个min分支求max
            for pAction in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, pAction), currentDepth, 1))
            return v

        def min_value(state, currentDepth, ghostNum):
            # 若当前状态已经赢了或输了
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # 初始化v
            v = float('Inf')
            # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
            for pAction in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents() - 1:
                    # 所有Ghost的min找完了 开始找下一个max
                    v = min(v, max_value(state.generateSuccessor(ghostNum, pAction), currentDepth))
                else:
                    # 继续下一个Ghost
                    v = min(v, min_value(state.generateSuccessor(ghostNum, pAction), currentDepth, ghostNum + 1))
            return v

            # pacman下一个状态可能的行动

        Pacman_Actions = gameState.getLegalActions(0)

        maximum = float('-Inf')
        result = ''

        # 针对下一个状态 寻找获胜概率最高的move
        for action in Pacman_Actions:
            if (action != "Stop"):
                currentDepth = 0
                # 而所有的Ghost希望胜利概率最低的选择
                currentMax = min_value(gameState.generateSuccessor(0, action), currentDepth, 1)
                if currentMax > maximum:
                    maximum = currentMax
                    result = action
        return result

        #util.raiseNotDefined()

"""Q3:alpha-beta剪枝"""
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, alpha, beta, currentDepth):
            # 当前深度加一
            currentDepth = currentDepth + 1
            # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            v = float('-Inf')
            # 对每个min分支求max
            for pAction in state.getLegalActions(0):
                if pAction != "Stop":
                    v = max(v, min_value(state.generateSuccessor(0, pAction), alpha, beta, currentDepth, 1))
                    # 若已经比beta要大了 就没有搜索下去的必要了
                    if v >= beta:
                        return v
                    # 更新alpha的值
                    alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, currentDepth, ghostNum):
            # 若当前状态已经赢了或输了
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # 初始化v
            v = float('Inf')
            # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
            for pAction in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents() - 1:
                    # 所有Ghost的min找完了 开始找下一个max
                    v = min(v, max_value(state.generateSuccessor(ghostNum, pAction), alpha, beta, currentDepth))
                else:
                    # 继续下一个Ghost
                    v = min(v,
                            min_value(state.generateSuccessor(ghostNum, pAction), alpha, beta, currentDepth,
                                      ghostNum + 1))
                # 若比alpha还要小了 就没搜索的必要了
                if v <= alpha:
                    return v
                # 更新beta的值
                beta = min(beta, v)
            return v

        # pacman下一个状态可能的行动
        pacmanActions = gameState.getLegalActions(0)
        maximum = float('-Inf')
        # 初始化alpha bate
        alpha = float('-Inf')
        beta = float('Inf')
        maxAction = ''

        # 针对下一个状态 寻找获胜概率最高的move
        for action in pacmanActions:
            if action != "Stop":
                currentDepth = 0
                # 而所有的Ghost希望胜利概率最低的选择
                currentMax = min_value(gameState.generateSuccessor(0, action), alpha, beta, currentDepth, 1)
                if currentMax > maximum:
                    maximum = currentMax
                    maxAction = action
        print
        maximum
        return maxAction

        #util.raiseNotDefined()
"""Q4:Expectimax搜索"""
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numb = gameState.getNumAgents()
        dep = self.depth        
        origin_move_ = "Stop"
        def expecti_max(state, depth=0, index=0, n=numb, depth_=dep, origin_move=origin_move_):
            if depth == depth_ or state.isWin() or state.isLose():
                return self.evaluationFunction(state),
            if index == 0:
                max_value = -999999.0
                move_ = origin_move
                for action in state.getLegalActions(0):
                    value = expecti_max(state.generateSuccessor(0, action), depth, (index + 1) % n)[0]
                    if value > max_value:
                        max_value = value
                        move_ = action
                return max_value, move_
            else:
                if index == n - 1:
                    depth += 1
                total = 0.0
                for action in state.getLegalActions(index):
                    value = expecti_max(state.generateSuccessor(index, action), depth, (index + 1) % n)[0]
                    total += float(value)
                average_value = total/float(len(state.getLegalActions(index)))
                return average_value, origin_move
        result = expecti_max(gameState)
        return result[1]
        #util.raiseNotDefined()
"""Q5:最佳估值搜索"""
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # 获取food所在的所有wall
    wallList = currentGameState.getWalls().asList()
    Pos = currentGameState.getPacmanPosition()
    # 获取食物在图中的分布（二维数组，有失误为T没食物为F)
    curFood = currentGameState.getFood()
    # 获取Ghost的位置
    GhostStates = currentGameState.getGhostStates()
    # 获取吃超级豆子之后 Ghost害怕还剩余的时间
    scaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    # 对这个选择评估的分数
    currscore = 0
    # 根据Ghost所在的位置，获取与当前位置的距离
    ghost_distances = []
    for gs in GhostStates:
        ghost_distances += [manhattanDistance(gs.getPosition(), Pos)]
    ghost_index = 0
    min_ghost_distances = min(ghost_distances)
    is_scared = False
    for time in scaredTimes:
        if time != 0:
            is_scared = True
        else:
            is_scared = False
            break
    # 获取food所在的所有pos
    curfoodList = curFood.asList()
    # 保存food的距离
    food_distences = []
    # 获取所有食物到达当前位置的距离
    for foodpos in curfoodList:
        food_distences += [manhattanDistance(Pos, foodpos)]
    # 对食物的距离取反
    inverse_food_distences = 0;
    if len(food_distences) > 0 and min(food_distences) > 0:
        inverse_food_distences = 1.0 / min(food_distences)

    if is_scared and min_ghost_distances != 0:
        # if min_ghost_distances < 10:
        # min_ghost_distances = 800 min_ghost_distances
        # else:
        # min_ghost_distances = 600 min_ghost_distances
        #print("Ghost Scared!")
        min_ghost_distances = min_ghost_distances * 0.8
    # 考虑了ghost与当前的距离，其权值更大
    if min(ghost_distances) == 0:
        currscore += inverse_food_distences
    else:
        currscore += min_ghost_distances * (float(inverse_food_distences))
    # 获取当前系统判定的分数 又可能当前吃到了豆子 分数更高些
    currscore += currentGameState.getScore()
    return currscore

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

