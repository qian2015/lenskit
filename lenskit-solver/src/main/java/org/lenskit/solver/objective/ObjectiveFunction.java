package org.lenskit.solver.objective;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public interface ObjectiveFunction {
    public void wrapOracle(StochasticOracle orc);
}