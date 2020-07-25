


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


TRUE = 1
FALSE  = 0
FORCED_ATTACK_TICK = 20
FORCED_DEFEND_TICK = 4
SAFE_DISTANCE = 5  #与怪兽的安全距离
SAFE_FODD_REMAIN = 4
#权重
WEIGHT_SCORE = 200
WEIGHT_FOOD = -5
WEIGHT_GHOST_SCARED = 0
WEIGHT_GHOST_NORMAL = 210
WEIGHT_FORCEDATTACK = 3000
WEIGHT_FORCEDBACK = 0
FORCED_AVOID_STUCK = 1
HOMELESS_TICK = 80
MAX_GREEDY = 5


class ReflexCaptureAgent(CaptureAgent):
  

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    #init
    features = util.Counter()
    successor = self.getSuccessor(gameState, action) # get the successor
    myPos = successor.getAgentState(self.index).getPosition() # get the successor pos
    foodList = self.getFood(successor).asList() # get the foodlist
    features['successorScore'] = self.getScore(successor) # set score feature

    if successor.getAgentState(self.index).isPacman:
      features['forcedOffensive'] = TRUE
    else:
      features['forcedOffensive'] = FALSE
    
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      #myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    # 考虑对手怪兽
    threateningGhostsPos = []
    distancesToGhosts = []
    #对手怪兽位置
    opponentsIndices = self.getOpponents(successor)
    for opponentIndex in opponentsIndices:
      oppent = successor.getAgentState(opponentIndex)
      if not oppent.isPacman and oppent.getPosition() != None:
        oppentPos = oppent.getPosition()
        threateningGhostsPos.append(oppentPos)

    # If there are some opponent ghosts
    for ghostPos in threateningGhostsPos:
      distancesToGhosts.append(self.getMazeDistance(myPos,ghostPos))

    if len(distancesToGhosts) > 0:
      minDisToGhost = min(distancesToGhosts)
      if minDisToGhost < SAFE_DISTANCE:
        features['distanceToGhost'] = minDisToGhost + features['successorScore'] 
      else:
        features['distanceToGhost'] = 0

    return features

  
  def getWeights(self, gameState, action):

    if self.forcedAttack:
      if self.forcedBack == 0:
          return {'forcedOffensive':WEIGHT_FORCEDATTACK,
                  'successorScore': WEIGHT_SCORE, 
                  'distanceToFood': WEIGHT_FOOD,
                  'distancesToGhost':WEIGHT_GHOST_NORMAL}
      else:
          return {'forcedOffensive':WEIGHT_FORCEDBACK,
                  'successorScore': WEIGHT_SCORE, 
                  'distanceToFood': WEIGHT_FOOD,
                  'distancesToGhost':WEIGHT_GHOST_NORMAL}

    successor = self.getSuccessor(gameState, action) # get the successor
    myPos = successor.getAgentState(self.index).getPosition() # get the successor pos

    # 考虑对手怪兽
    minDistance = 10000000
    scaredGhost = []
    ghostScared = False
    # 对手怪兽位置
    opponentsIndices = self.getOpponents(successor)
    for opponentIndex in opponentsIndices:
      oppent = successor.getAgentState(opponentIndex)
      if not oppent.isPacman and oppent.getPosition() != None:
        oppentPos = oppent.getPosition()
        disToOppent = self.getMazeDistance(myPos,oppentPos)
        if disToOppent < minDistance:
          minDistance = disToOppent
          scaredGhost.append(oppent)
    if len(scaredGhost) > 0:
      if scaredGhost[-1].scaredTimer > 0:
        ghostScared = True

    if ghostScared:
      weightGhost = WEIGHT_GHOST_SCARED
    else:
      weightGhost = WEIGHT_GHOST_NORMAL

    return {'forcedOffensive':WEIGHT_FORCEDBACK,'successorScore': WEIGHT_SCORE, 'distanceToFood': WEIGHT_FOOD,'distancesToGhost':weightGhost}

  def getOpponentPositions(self, gameState):
    return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

  def randomChooseOneDesirableAction(self,simulatedState):
    actionsBase = simulatedState.getLegalActions(self.index)
    actionsBase.remove(Directions.STOP)
    if len(actionsBase) == 1: 
      return actionsBase[0]
    else: 
      backwardsDirection = Directions.REVERSE[simulatedState.getAgentState(self.index).configuration.direction]
      if backwardsDirection in actionsBase:
       actionsBase.remove(backwardsDirection)
      return random.choice(actionsBase)
      
  def monteCarloSimulation(self,gameState,rounds):
    simulatedState = gameState.deepCopy()
    while rounds > 0:
      simulatedAction = self.randomChooseOneDesirableAction(simulatedState)
      simulatedState = simulatedState.generateSuccessor(self.index,simulatedAction)
      rounds = rounds - 1
    return self.evaluate(simulatedState,Directions.STOP)

  
  def filterOutUndesirableActaions(self,gameState,action,rounds):
    if rounds == 0:
      return True
    
    newState = gameState.generateSuccessor(self.index,action)
    currentScore = self.getScore(gameState)
    newScore = self.getScore(newState)
    currentFoodList = self.getFood(gameState).asList()
    newFoodList = self.getFood(newState).asList()
    if currentScore < newScore:
      return True

    actionsBase = newState.getLegalActions(self.index)
    actionsBase.remove(Directions.STOP)
    towardsDirection = newState.getAgentState(self.index).configuration.direction
    backwardsDirection = Directions.REVERSE[towardsDirection]
    if backwardsDirection in actionsBase:
      if len(currentFoodList) == len(newFoodList):
       actionsBase.remove(backwardsDirection)
      else:
       return True

    if len(actionsBase) == 0:
      return False

    for action in actionsBase:
      if self.filterOutUndesirableActaions(newState,action,rounds-1):
        return True
    return False
        
  def filterOutRiskyActions(self,gameState,action,rounds):
    if rounds == 0:
      return True

    newState = gameState.generateSuccessor(self.index,action)
    currentScore = self.getScore(gameState)
    newScore = self.getScore(newState)
    if currentScore < newScore:
        return True

    actionsBase = newState.getLegalActions(self.index)
    actionsBase.remove(Directions.STOP)
    towardsDirection = newState.getAgentState(self.index).configuration.direction
    backwardsDirection = Directions.REVERSE[towardsDirection]

    if backwardsDirection in actionsBase:
        actionsBase.remove(backwardsDirection)
    else:
        return True
    
    if len(actionsBase) == 0:
        return False

    for action in actionsBase:
        if self.filterOutRiskyActions(newState,action,rounds-1):
            return True
    return False
        
  def __init__(self, index):
        CaptureAgent.__init__(self, index)
        
        self.currentFoodSize = 10000000
        self.prev1Pos = (-1,-1)
        self.prev2Pos = (-2,-2)
        self.prev3Pos = (-3,-3)
        self.prev4Pos = (-4,-4)
        self.myPos = (-5,-5)
        self.tick = 0
        self.forcedAttack = False
        self.lastTickFoodList = []
        self.currentTickFoodList = []
        self.forcedBack = 0
        self.isHomeless = 0
        self.homeTarget = None
        self.ifStuckList = []
        self.switchTargetMode = False
        self.modeTarget = None
        self.foodFastEaten = 0
        self.firstAttackArea = []
        self.startLock = 0
        self.currentCapsuleSize = 0
        self.lastCapsuleSize = 0

  def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        self.initPosition = gameState.getAgentState(self.index).getPosition()
        self.setFirstAttackArea(gameState)
        
  def getLayoutInfo(self,gameState):
    layoutInfo = []
    layoutWidth = gameState.data.layout.width
    layoutHeight = gameState.data.layout.height
    layoutCentralX = int((layoutWidth - 2) / 2)
    if not self.red:
      layoutCentralX +=1
    layoutCentralY = int((layoutHeight - 2)/ 2)
    layoutInfo.extend((layoutWidth,layoutHeight,layoutCentralX,layoutCentralY))
    return layoutInfo

  def setFirstAttackArea(self,gameState):
        layoutInfo = self.getLayoutInfo(gameState)
        self.firstAttackArea = []
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.firstAttackArea.append((layoutInfo[2], i))
        while len(self.firstAttackArea) > 2:
          self.firstAttackArea.remove(self.firstAttackArea[0])
          self.firstAttackArea.remove(self.firstAttackArea[-1])
        if  len(self.firstAttackArea) == 2:
          self.firstAttackArea.remove(self.firstAttackArea[0])

  def isForcedAttackRequired (self,gameState):
        remainingFoodList = self.getFood(gameState).asList()
        remainingFoodSize = len(remainingFoodList)

        if remainingFoodSize == self.currentFoodSize:
            self.tick = self.tick + 1
        else:
            self.currentFoodSize = remainingFoodSize
            self.tick = 0
        if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
            self.tick = 0
        if self.tick > FORCED_ATTACK_TICK:
            return True
        else:
            return False
  def isForcedAvoidStuck (self,gameState):
        sum = 0
        self.myPos = gameState.getAgentState(self.index).getPosition()
        if len(self.ifStuckList) > 9:
            self.ifStuckList.pop(0)
        if self.myPos == self.prev2Pos and self.myPos == self.prev4Pos:
            if self.prev1Pos == self.prev3Pos:
                #danbu zhendang
                self.ifStuckList.append(1)
            else:
                #zhijiao zhendang
                self.ifStuckList.append(1)
        else:
            self.ifStuckList.append(0)
        self.prev4Pos = self.prev3Pos
        self.prev3Pos = self.prev2Pos
        self.prev2Pos = self.prev1Pos
        self.prev1Pos = self.myPos
        if len(self.ifStuckList) < 9:
            return False
        else:
            for i in range(len(self.ifStuckList)):
                sum += self.ifStuckList[i]
            if sum > FORCED_AVOID_STUCK:
                self.switchTargetMode = True
                return True
            else:
                return False

  def chooseAction(self, gameState):

    self.myPos = gameState.getAgentState(self.index).getPosition()
        
    if self.myPos == self.initPosition:
            #todo go to the target
            self.startLock = 1
        
    if self.myPos == self.firstAttackArea[0]:
            self.startLock = 0
            
    if self.startLock == 1:
            candidateActions = gameState.getLegalActions(self.index)
            candidateActions.remove(Directions.STOP)
            goodActions = []
            fvalues = []
        
            for a in candidateActions:
              new_state = gameState.generateSuccessor(self.index, a)
              newpos = new_state.getAgentPosition(self.index)
              goodActions.append(a)
              fvalues.append(self.getMazeDistance(newpos, self.firstAttackArea[0]))
        
            best = min(fvalues)
            bestActions = [a for a, v in zip(goodActions, fvalues) if v == best]
            bestAction = random.choice(bestActions)
            return bestAction
        
    if self.startLock==0:
            
        self.currentTickFoodList = self.getFood(gameState).asList()
        self.currentCapsuleSize = len(self.getCapsules(gameState))
        realLastCapsuleLen = self.lastCapsuleSize
        
        realLastFoodLen = len(self.lastTickFoodList)
        
        if len(self.currentTickFoodList) < len(self.lastTickFoodList):
            self.forcedBack = 1
        self.lastTickFoodList = self.currentTickFoodList
        self.lastCapsuleSize = self.currentCapsuleSize
        
        if not gameState.getAgentState(self.index).isPacman:
            self.forcedBack = 0
            self.isHomeless = 0
        else:
            self.isHomeless += 1
        
        self.forcedAttack = self.isForcedAttackRequired(gameState)
        
        actionsBase = gameState.getLegalActions(self.index)
        actionsBase.remove(Directions.STOP)
        
        minDistance = float("inf")
        
        opponentsIndices = self.getOpponents(gameState)
        for opponentIndex in opponentsIndices:
            oppent = gameState.getAgentState(opponentIndex)
            if not oppent.isPacman and oppent.getPosition() != None and not oppent.scaredTimer > 0 :
                oppentPos = oppent.getPosition()
                disToOppent = self.getMazeDistance(self.myPos,oppentPos)
                if disToOppent < minDistance:
                   minDistance = disToOppent

        
  
        candidateActions = []
        for a in actionsBase:
            if minDistance > 3:
              if self.filterOutUndesirableActaions(gameState, a, 6):
                candidateActions.append(a)
            else:
              if self.filterOutRiskyActions(gameState,a,9):
                candidateActions.append(a)

               
        self.isForcedAvoidStuck(gameState)
        if self.currentCapsuleSize < realLastCapsuleLen:
           self.switchTargetMode = True
           self.foodFastEaten = 0
        if minDistance <= 5:
           self.switchTargetMode = False
        if (len(self.currentTickFoodList) < len (self.lastTickFoodList)):
           self.switchTargetMode = False
          
        if self.switchTargetMode: 
            if not gameState.getAgentState(self.index).isPacman:
                self.foodFastEaten = 0
            
            modeMinDistance = float("inf")
            
            if len(self.currentTickFoodList) < realLastFoodLen:
                self.foodFastEaten += 1
            
            if len(self.currentTickFoodList)==0 or self.foodFastEaten >= MAX_GREEDY:
              self.modeTarget = self.initPosition
            else:
             for food in self.currentTickFoodList:
               distance = self.getMazeDistance(self.myPos,food)
               if distance < modeMinDistance:
                 modeMinDistance = distance
                 self.modeTarget = food

            candidateActions = gameState.getLegalActions(self.index)
            candidateActions.remove(Directions.STOP)
            goodActions = []
            fvalues = []
        
            for a in candidateActions:
              new_state = gameState.generateSuccessor(self.index, a)
              newpos = new_state.getAgentPosition(self.index)
              goodActions.append(a)
              fvalues.append(self.getMazeDistance(newpos, self.modeTarget))
        
            best = min(fvalues)
            bestActions = [a for a, v in zip(goodActions, fvalues) if v == best]
            bestAction = random.choice(bestActions)
            return bestAction

        else:
          self.foodFastEaten = 0
          fvalues = []
          for a in candidateActions:
              new_state = gameState.generateSuccessor(self.index, a)
              value = 0
              for i in range(1, 24):
                  value += self.monteCarloSimulation(new_state,12)
              fvalues.append(value)

          best = max(fvalues)
          bestActions = [a for a, v in zip(candidateActions, fvalues) if v == best]
          bestAction = random.choice(bestActions)
        return bestAction

class DefensiveReflexAgent(ReflexCaptureAgent):
  def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.lastTickFoodList = []
        self.isFoodEaten = False
        self.patrolDict = {}
        self.tick = 0
        self.gazeboDict = {}
  
  def getLayoutInfo(self,gameState):
    
    layoutInfo = []
    layoutWidth = gameState.data.layout.width
    layoutHeight = gameState.data.layout.height
    layoutCentralX = int((layoutWidth - 2) / 2)
    if not self.red:
      layoutCentralX +=1
    layoutCentralY = int((layoutHeight - 2)/ 2)
    layoutInfo.extend((layoutWidth,layoutHeight,layoutCentralX,layoutCentralY))
    return layoutInfo

  def setDefensiveArea(self,gameState):

        layoutInfo = self.getLayoutInfo(gameState)

        self.coreDefendingArea = []
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.coreDefendingArea.append((layoutInfo[2], i))

        desiredSize = layoutInfo[3]
        currentSize = len(self.coreDefendingArea)
        
        while desiredSize < currentSize:

          self.coreDefendingArea.remove(self.coreDefendingArea[0])
          self.coreDefendingArea.remove(self.coreDefendingArea[-1])
          currentSize = len(self.coreDefendingArea)
        while len(self.coreDefendingArea) > 2:
        #for i in range(currentSize/4):
          self.coreDefendingArea.remove(self.coreDefendingArea[0])
          self.coreDefendingArea.remove(self.coreDefendingArea[-1])
        #if  len(self.coreDefendingArea) == 2:
          #self.coreDefendingArea.remove(self.coreDefendingArea[0])
         
  def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        self.setDefensiveArea(gameState)  
  def isForcedDefendRequired(self,gameState):
        candidateActions = []
        actions = gameState.getLegalActions(self.index)
        reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)
        if reversed_direction in actions:
            actions.remove(reversed_direction)
            
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman:
                candidateActions.append(a)
                
        if len(candidateActions) == 0:
            self.tick = 0
        else:
            self.tick = self.tick + 1
        
        if self.tick > FORCED_DEFEND_TICK or self.tick == 0:
            candidateActions.append(reversed_direction)
            
        return candidateActions
                                            
  def chooseAction(self, gameState):

        currentTickFoodList = self.getFoodYouAreDefending(gameState).asList()

        mypos = gameState.getAgentPosition(self.index)
        if mypos == self.target:
            self.target = None
        # Get the cloest invader's position and set target as invader
        threateningInvaderPos = []
        cloestInvaders = []
        minDistance = float("inf")
        
        opponentsIndices = self.getOpponents(gameState)
        for opponentIndex in opponentsIndices:
          oppent = gameState.getAgentState(opponentIndex)
          if oppent.isPacman and oppent.getPosition() != None:
            oppentPos = oppent.getPosition()
            threateningInvaderPos.append(oppentPos)

        if len (threateningInvaderPos) > 0:
          for position in threateningInvaderPos:
            distance = self.getMazeDistance(position,mypos)
            if distance < minDistance:
              minDistance = distance
              cloestInvaders.append(position)
          self.target = cloestInvaders[-1]

        # get the eaten food position
        else:
          if len(self.lastTickFoodList) > 0 and len(currentTickFoodList) < len(self.lastTickFoodList) :
            eatenFood = set(self.lastTickFoodList) - set(currentTickFoodList)
            
            self.target = eatenFood.pop()

        self.lastTickFoodList = currentTickFoodList
        
        if self.target == None:
          if len(currentTickFoodList) <= SAFE_FODD_REMAIN:
            highPriorityFood = currentTickFoodList + self.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(highPriorityFood)
          else:
            self.target = random.choice(self.coreDefendingArea)
        # evaluates candiateActions and get the best 
        candidateActions = self.isForcedDefendRequired(gameState)
        goodActions = []
        fvalues = []
        
        for a in candidateActions:
            new_state = gameState.generateSuccessor(self.index, a)
            newpos = new_state.getAgentPosition(self.index)
            goodActions.append(a)
            fvalues.append(self.getMazeDistance(newpos, self.target))
        
        best = min(fvalues)
        bestActions = [a for a, v in zip(goodActions, fvalues) if v == best]
        bestAction = random.choice(bestActions)
        return bestAction
