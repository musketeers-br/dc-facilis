import sys
from crew import TripAdvisorCrew

def run():
    inputs = {
        "range": "June 2025",
        "origin": "San Francisco",
        "city": "Paris",
        "interests": "art",
    }
    TripAdvisorCrew().crew().kickoff(inputs=inputs)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "range": "June 2025",
        "origin": "San Francisco",
        "city": "Paris",
        "interests": "art",
    }
    try:
        TripAdvisorCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")