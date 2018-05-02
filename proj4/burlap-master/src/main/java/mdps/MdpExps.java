package mdps;

import java.util.List;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyDeterministicQPolicy;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

public class MdpExps {
	
	GridWorldDomain gw;
	GridWorldTerminalFunction tf;
	GridWorldRewardFunction rf;
	OOSADomain domain;
	GridWorldState initialState;
	SimpleHashableStateFactory hashingFactory;
	StateConditionTest goalCondition;
	SimulatedEnvironment env;
	
	public MdpExps(int p) {
		if (p==0) createSimpleGrid();
		else createLargeGrid();
	}

	private void createSimpleGrid() {
        int width, height;
	      int[][] map = new int[][]{
				{0,0,0},
				{0,1,0},
				{0,0,0},
				{0,0,0}
	      };
        gw = new GridWorldDomain(map);
        width = gw.getWidth();
        height = gw.getHeight();
        gw.setProbSucceedTransitionDynamics(0.8);
        
        tf = new GridWorldTerminalFunction(width-1, height-1);
        tf.markAsTerminalPosition(width-1, height-2);
        rf = new GridWorldRewardFunction(width, height, -1.0);
        rf.setReward(width-1, height-1, 10);
        rf.setReward(width-1, height-2, -10);
        gw.setTf(tf);
        gw.setRf(rf);

        domain = gw.generateDomain();
        initialState = new GridWorldState(
        		new GridAgent(0, 0),
                new GridLocation(width-1, height-1, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);
	}
	
	private void createLargeGrid() {
        int width, height;
        int[][] map = new int[][]{
			{0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0},
			{0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0},
			{0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1,0},
			{0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        };
        gw = new GridWorldDomain(map);
        width = gw.getWidth();
        height = gw.getHeight();
        //System.out.println(width+"*"+height);
        gw.setProbSucceedTransitionDynamics(0.8);
        
        tf = new GridWorldTerminalFunction(width-1, height-1);
        tf.markAsTerminalPosition(width-1, height-2);
        rf = new GridWorldRewardFunction(width, height, -1.0);
        rf.setReward(width-1, height-1, 10);
        rf.setReward(width-1, height-2, -10);
        gw.setTf(tf);
        gw.setRf(rf);

        domain = gw.generateDomain();
        initialState = new GridWorldState(
        		new GridAgent(0, 0),
                new GridLocation(width-1, height-1, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);
	}
	
    public void valueIterationAnalysis(String mdp, int minIter, int maxIter, int interval){
        double planningTime = 0;
        ValueIteration planner = null;
        Policy policy = null;
        Episode ep = null;

        System.out.println("Value Iteration data:");
        for (int i = minIter; i <= maxIter; i+=interval){
            long startTime = System.nanoTime();
            planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, i);
            policy = planner.planFromState(initialState);
            planningTime = (System.nanoTime() - startTime) / 1e6; // in millisec
            //System.out.println("time: " + planningTime);
            ep = PolicyUtils.rollout(policy, initialState, domain.getModel());
        }

        System.out.println("Value Iteration:");
        System.out.println("time: " + planningTime);
        System.out.println("Steps in episode: " + ep.maxTimeStep() + ", Total return:" + ep.discountedReturn(1.0));
        System.out.println("\n");

        if (!mdp.isEmpty()) {
        	visualizeValueFunction(planner, policy, planner.totalIterationNumber+" Value Iterations on "+mdp);
        }
    }
    
    public void policyIterationAnalysis(String mdp, int minIter, int maxIter, int interval){
        double planningTime = 0;
        PolicyIteration planner = null;
        Policy policy = null;
        Episode ep = null;

        System.out.println("Policy Iteration data:");
        for (int i = minIter; i <= maxIter; i+=interval){
            long startTime = System.nanoTime();
            planner = new PolicyIteration(domain, 0.99, hashingFactory, 0.001, i, i);
            policy = planner.planFromState(initialState);
            planningTime = (System.nanoTime() - startTime) / 1e6; // in millisec
            //System.out.println(i+"th iteration takes time: " + planningTime);
            ep = PolicyUtils.rollout(policy, initialState, domain.getModel());
        }

        System.out.println("Value Iteration:");
        System.out.println("time: " + planningTime);
        System.out.println("Steps in episode: " + ep.maxTimeStep() + ", Total return:" + ep.discountedReturn(1.0));
        System.out.println("\n");

        if (!mdp.isEmpty()) {
        	visualizeValueFunction(planner, policy, planner.getTotalPolicyIterations()+" Policy Iterations on "+mdp);
        }
    }
    
    public void qLearningEpisodicAnalysis(String mdp, int numOfEps){
        EpisodeData ed = new EpisodeData();
        int numOfTries = 100;
        for (int t = 0; t < numOfTries; t++) {
            QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.8);
            for (int i = 0; i < numOfEps; i++) {
                Episode e = agent.runLearningEpisode(env);
                env.resetEnvironment();
                if (t == 0)
                    ed.addTuple(i, e.discountedReturn(1.0), e.maxTimeStep(), 0);
                else {
                    double accumReward = ed.rewards.get(i);
                    accumReward += e.discountedReturn(1.0);
                    if (t == numOfTries-1)
                    	accumReward = accumReward / numOfTries;
                    ed.rewards.set(i, accumReward);
                }
            }
            if (t == numOfTries-1) {
                visualizeValueFunction(agent, new GreedyQPolicy(agent), numOfEps+" episodes of Qlearning on "+mdp);
                ed.write(mdp + "_qlearning_eps.csv");
            }
        }
    }
    
    public void qLearningPlanningAnalysis(String mdp, int numOfEps){
        long startTime = System.nanoTime();
        QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.8);
        agent.setMaximumEpisodesForPlanning(numOfEps);
        agent.setMaxQChangeForPlanningTerminaiton(0.001);
        //System.out.println("Planning");
        Policy policy = agent.planFromState(initialState);
        double planningTime = (System.nanoTime() - startTime) / 1e6; // in millisec
        Episode ep = PolicyUtils.rollout(policy, initialState, domain.getModel());
        System.out.println("QLearning Convergence Test:");
        System.out.println("time: " + planningTime);
        System.out.println("Steps in episode: " + ep.maxTimeStep() + ", Total return:" + ep.discountedReturn(1.0));
        System.out.println("\n");
        
        if (!mdp.isEmpty()) {
            visualizeValueFunction(agent, new GreedyQPolicy(agent), agent.totalNumOfEpisodes+" episodes of converging Qlearning on "+mdp);
        }
            
    }
    
    public void qLearningPolicyAnalysis(String mdp, int numOfEps, double epsilon, String learningPolicy){
        long startTime = System.nanoTime();
        QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.8);
        agent.setMaximumEpisodesForPlanning(numOfEps);
        agent.setMaxQChangeForPlanningTerminaiton(0.001);
        Policy learning = null;
        String visualTitle = null;
        if (learningPolicy.equals("Epsilon")) {
        	learning = new EpsilonGreedy(agent, epsilon);
        	visualTitle = " episodes of Epsilon "+ epsilon +" Greedy Qlearning on "+mdp;
        }
        else if (learningPolicy.equals("Random")) {
        	learning = new RandomPolicy(domain);        	
        	visualTitle = " episodes of RandomPolicy Qlearning on "+mdp;
        }
        else {
        	learning = new GreedyDeterministicQPolicy(agent);
        	visualTitle = " episodes of GreedyDeterministicQPolicy Qlearning on "+mdp;
        }
        agent.setLearningPolicy(learning);
        
        Policy policy = agent.planFromState(initialState);
        double planningTime = (System.nanoTime() - startTime) / 1e6; // in millisec
        Episode ep = PolicyUtils.rollout(policy, initialState, domain.getModel());
        System.out.println("QLearning Learning Policy Tests:");
        System.out.println("time: " + planningTime);
        System.out.println("Steps in episode: " + ep.maxTimeStep() + ", Total return:" + ep.discountedReturn(1.0));
        System.out.println("\n");
        if (!mdp.isEmpty()) {
            visualizeValueFunction(agent, new GreedyQPolicy(agent), agent.totalNumOfEpisodes+visualTitle);
        }
            
    }
	
	private void visualizeGrid() {
		Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
		VisualExplorer exp = new VisualExplorer(domain, v, initialState);
		exp.initGUI();
	}
	
    private void visualizeValueFunction(ValueFunction valueFunction, Policy policy, String title){
        List<State> reachableStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);
        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(reachableStates, gw.getWidth(), gw.getHeight(), valueFunction, policy);
        gui.setTitle(title);
        gui.initGUI();
    }
	
	public static void main(String[] args) {
		// First mdp
		String mdp = "SimpleGrid";
		MdpExps simpleMdpExps = new MdpExps(0); 
		int numOfEps = 100;
		
		simpleMdpExps.visualizeGrid();
		simpleMdpExps.valueIterationAnalysis(mdp, 1, 100, 5);
		simpleMdpExps.policyIterationAnalysis(mdp, 1, 100, 5);
		
		simpleMdpExps.qLearningEpisodicAnalysis(mdp, 500);
		simpleMdpExps.qLearningPlanningAnalysis(mdp, Integer.MAX_VALUE);
		simpleMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0.6, "Epsilon");
		simpleMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0.2, "Epsilon");
		simpleMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 1, "Random");
		simpleMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0, "GreedyDeterministicQ");
		
		
		// Second mdp
		mdp = "LargeGrid";
		MdpExps largeMdpExps = new MdpExps(1); 
		numOfEps = 200;
		
		largeMdpExps.visualizeGrid();
		largeMdpExps.valueIterationAnalysis(mdp, 1, 100, 5);
		largeMdpExps.policyIterationAnalysis(mdp, 1, 100, 5);
		
		largeMdpExps.qLearningEpisodicAnalysis(mdp, 500);
		largeMdpExps.qLearningPlanningAnalysis(mdp, Integer.MAX_VALUE);
		largeMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0.6, "Epsilon");
		largeMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0.2, "Epsilon");
		largeMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 1, "Random");
		largeMdpExps.qLearningPolicyAnalysis(mdp, numOfEps, 0, "GreedyDeterministicQ");
	}

}
