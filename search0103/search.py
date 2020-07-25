# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
"""#Q1:深度优先算法"""
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()  # 使用优先队列，每次扩展都是选择当前代价最小的方向，即队头
    fringe.push((problem.getStartState(), []))  # 把初始化点加入队列，开始扩展
    closed = []  # 标记已经走过的点
    while fringe:
        cur_node, actions = fringe.pop()  # 当前状态
        if problem.isGoalState(cur_node):
            # 返回到达终点的操作顺序
            return actions
        if cur_node not in closed:
            closed.append(cur_node)
            expand = problem.getSuccessors(cur_node)
            for position, direction, cost in expand:
                tempActions = actions + [direction]
                if position not in closed:
                    fringe.push((position, tempActions))


    util.raiseNotDefined()
"""#Q2:宽度优先算法"""
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()  # 使用优先队列，每次扩展都是选择当前代价最小的方向，即队头
    fringe.push((problem.getStartState(), []))  # 把初始化点加入队列，开始扩展
    closed = []  # 标记已经走过的点
    while fringe:
        cur_node, actions = fringe.pop()  # 当前状态
        if problem.isGoalState(cur_node):
            # 返回到达终点的操作顺序
            return actions
        if cur_node not in closed:
            closed.append(cur_node)
            expand = problem.getSuccessors(cur_node)
            for position, direction, cost in expand:
                tempActions = actions + [direction]
                if position not in closed:
                    fringe.push((position, tempActions))

    util.raiseNotDefined()

"""#Q3:代价一致算法"""
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # 使用优先队列，每次扩展都是选择当前代价最小的方向，即队头
    fringe.push((problem.getStartState(), []),0)  # 把初始化点加入队列，开始扩展
    closed = []  # 标记已经走过的点
    while fringe:
        currState, actions = fringe.pop()  # 当前状态
        if problem.isGoalState(currState):
            return actions
        if currState not in closed:
            closed.append(currState)
            successors = problem.getSuccessors(currState)
            for successor, action, cost in successors:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + cost  # 对可选的几个方向，计算代价
                if successor not in closed:
                    fringe.push((successor, tempActions), nextCost)

    #尝试简洁语言
    """moves_priority_queue = util.PriorityQueue()
    moves_priority_queue.push([], 0)
    closed = []
    fringe = util.PriorityQueue()
    # 加入起始状态节点
    fringe.push(problem.getStartState(), 0)
    while not fringe.isEmpty():
        moves = moves_priority_queue.pop()
        cur_node = fringe.pop()
        if cur_node not in closed:
            # 当前节点加入到走过路程中
            closed.append(cur_node)
            if problem.isGoalState(cur_node):
                return moves
            for position, direction, f_value_per_move in problem.getSuccessors(cur_node):
                moves_priority_queue.push(moves+[direction],  problem.getCostOfActions(moves)+f_value_per_move)
                fringe.push(position,  problem.getCostOfActions(moves)+f_value_per_move)"""
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
"""#Q4:A* 算法,利用曼哈顿距离作为启发函数"""
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # 使用优先队列，每次扩展都是选择当前代价最小的方向，即队头
    actions = []  # 选择的操作
    fringe.push((problem.getStartState(), actions), 0)  # 把初始化点加入队列，开始扩展
    closed = []  # 标记已经走过的点
    while fringe:
        currState, actions = fringe.pop()  # 当前状态
        if problem.isGoalState(currState):
            break
        if currState not in closed:
            closed.append(currState)
            successors = problem.getSuccessors(currState)
            for successor, action, cost in successors:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + heuristic(successor, problem)  # 对可选的几个方向，计算代价
                if successor not in closed:
                    fringe.push((successor, tempActions), nextCost)
    return actions  # 返回到达终点的操作顺序

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
