package org.lenskit.solver.method;

import org.lenskit.solver.objective.LearningInstance;
import org.lenskit.solver.objective.LearningModel;
import org.lenskit.solver.objective.ObjectiveFunction;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class ExpectationMaximization implements OptimizationMethod {
    private int maxIter;
    private double tol;
    private OptimizationMethod method;
    
    public ExpectationMaximization() {
        maxIter = 50;
        tol = 1.0;
        method = new BatchGradientDescent();
    }

    public double minimize(LearningModel model, ObjectiveFunction objFunc) {
        ObjectiveTerminationCriterion termCrit = new ObjectiveTerminationCriterion(tol, maxIter);
        double objVal = 0;
        while (termCrit.keepIterate()) {
            objVal = 0;
            model.startNewIteration();
            LearningInstance ins;
            while ((ins = model.getLearningInstance()) != null) {
                objVal += model.expectation(ins);
            }
            LearningModel subModel = model.maximization();
            if (subModel != null) {
                objVal += method.minimize(subModel, objFunc);
            }
            termCrit.addIteration(objVal);
        }
        return objVal;
    }
}
