from src.boa import *



# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    # Add arguments
    parser.add_argument("training_dir", type=str, help="Directory to the training dataset.")
    parser.add_argument("testing_dir", type=str, help="Directory to the training dataset.")
    parser.add_argument("grammar_dir", type=str, help="Directory to the grammar.")
    parser.add_argument("-s", "--seed", type=int, help="The seed", default=1) 
    parser.add_argument("-m", "--metric", type=str, help="The metric to be used during biochemical property predicion optimisation procedure", default="auc")
    parser.add_argument("-e", "--exp_name", type=str, help="The name of the experiment", default="Exp_ADMET")
    parser.add_argument("-t", "--time", type=int, help="Time in minutes to run the method", default=5)
    
    # Parse arguments
    args = parser.parse_args()
    training_dir = args.training_dir
    testing_dir = args.training_dir
    grammar_dir = args.grammar_dir
    seed = args.seed
    metric = args.metric
    exp_name = args.exp_name
    max_time = args.time
    
    random.seed(seed)  # For reproducibility

    # Define grammar
    with open(grammar_dir, "r") as file:
        grammar_text = file.read()

    # Load grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run GGP
    ggp = GrammarBasedGP(grammar, training_dir, testing_dir, fitness_metric=metric, experiment_name=exp_name, seed=seed, max_time=max_time)
    best_program = ggp.evolve()
    
 
