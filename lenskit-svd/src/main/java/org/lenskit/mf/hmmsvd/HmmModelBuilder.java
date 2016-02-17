package org.lenskit.mf.hmmsvd;

import java.io.IOException;

import org.lenskit.solver.method.ExpectationMaximization;
import org.lenskit.solver.method.OptimizationMethod;
import org.lenskit.solver.objective.LogisticLoss;
import org.lenskit.solver.objective.ObjectiveFunction;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class HmmModelBuilder {
    private HmmModel model;
    private OptimizationMethod method;
    private ObjectiveFunction loss;

    public HmmModelBuilder(int numPos, HmmSVDFeatureInstanceDAO dao) {
        try {
            model = new HmmModel(numPos, dao);
            method = new ExpectationMaximization();
            loss = new LogisticLoss();
        } catch (IOException e) {}
    }

    public HmmModel build() throws IOException {
        model.assignVariables();
        method.minimize(model, loss);
        return model;
    }
}
