import csv
import itertools
import random


CARDS = (0, 1, 2)  # Jack, Queen, King
CARD_NAMES = ('J', 'Q', 'K')
ALL_DEALS = list(itertools.permutations(CARDS, 2))

FINISHED_HANDS = ('cc', 'bf', 'bk', 'cbf', 'cbk')
DECISION_POINTS = ('', 'c', 'b', 'cb')
ACTION_NAMES = {'c': 'check', 'b': 'bet', 'f': 'fold', 'k': 'call'}


def legal_actions(history):
    if history == 'b' or history == 'cb':
        return ('f', 'k')
    return ('c', 'b')


def hand_payoff(cards, history):
    """Return Player 1's net profit for a completed hand."""
    if history == 'bf':
        return 1
    if history == 'cbf':
        return -1

    pot_profit = 1 if history == 'cc' else 2
    if cards[0] > cards[1]:
        return pot_profit
    return -pot_profit


class DecisionNode:
    def __init__(self):
        self.regrets = [0.0, 0.0]
        self.strategy_total = [0.0, 0.0]

    def current_strategy(self):
        positive_regrets = [max(0.0, r) for r in self.regrets]
        total = sum(positive_regrets)

        if total == 0:
            return [0.5, 0.5]
        return [r / total for r in positive_regrets]

    def average_strategy(self):
        total = sum(self.strategy_total)
        if total == 0:
            return [0.5, 0.5]
        return [x / total for x in self.strategy_total]


class PokerSolver:
    def __init__(self):
        self.nodes = {}
        for card in CARDS:
            for history in DECISION_POINTS:
                self.nodes[(card, history)] = DecisionNode()

    def cfr(self, cards, history, reach_p1, reach_p2,
            regret_updates, strategy_updates):
        if history in FINISHED_HANDS:
            return hand_payoff(cards, history)

        player = len(history) % 2
        key = (cards[player], history)
        strategy = self.nodes[key].current_strategy()

        # Evaluate both possible actions before changing the strategy.
        action_values = []
        for i, action in enumerate(legal_actions(history)):
            next_p1 = reach_p1
            next_p2 = reach_p2

            if player == 0:
                next_p1 *= strategy[i]
            else:
                next_p2 *= strategy[i]

            result = self.cfr(cards, history + action, next_p1, next_p2,
                              regret_updates, strategy_updates)
            action_values.append(result)

        expected = 0.0
        for i in range(2):
            expected += strategy[i] * action_values[i]

        if player == 0:
            my_reach = reach_p1
            opponent_reach = reach_p2
            sign = 1
        else:
            my_reach = reach_p2
            opponent_reach = reach_p1
            sign = -1  # Payoffs above are always from Player 1's view.

        for i in range(2):
            regret = sign * (action_values[i] - expected)
            regret_updates[key][i] += opponent_reach * regret / 6
            strategy_updates[key][i] += my_reach * strategy[i] / 6

        return expected

    def train(self, iterations=10000):
        results = []

        for iteration in range(1, iterations + 1):
            regret_updates = {key: [0.0, 0.0] for key in self.nodes}
            strategy_updates = {key: [0.0, 0.0] for key in self.nodes}

            # Kuhn Poker only has six possible deals, so check all of them.
            for cards in ALL_DEALS:
                self.cfr(cards, '', 1.0, 1.0, regret_updates, strategy_updates)

            for key, node in self.nodes.items():
                for i in range(2):
                    node.regrets[i] += regret_updates[key][i]
                    node.strategy_total[i] += strategy_updates[key][i]

            if iteration == 1 or iteration % 500 == 0 or iteration == iterations:
                value, exploitability = evaluate(self.get_policy())
                results.append((iteration, value, exploitability))
                print(f'Iteration {iteration:5d} | EV {value:+.6f} | '
                      f'Exploitability {exploitability:.6f}')

        return results

    def get_policy(self):
        return {key: node.average_strategy() for key, node in self.nodes.items()}


def expected_value(player1_policy, player2_policy):
    def walk(cards, history):
        if history in FINISHED_HANDS:
            return hand_payoff(cards, history)

        player = len(history) % 2
        policy = player1_policy if player == 0 else player2_policy
        probabilities = policy[(cards[player], history)]

        value = 0.0
        for action, probability in zip(legal_actions(history), probabilities):
            value += probability * walk(cards, history + action)
        return value

    return sum(walk(cards, '') for cards in ALL_DEALS) / 6


def evaluate(policy):
    # Find each player's best possible response to the trained strategy.
    best_responses = []

    for player in (0, 1):
        keys = [key for key in policy if len(key[1]) % 2 == player]
        best_value = float('-inf')

        # Each player has 6 decision points with 2 actions: 2^6 = 64 policies.
        for choices in itertools.product((0, 1), repeat=len(keys)):
            candidate = dict(policy)
            for key, choice in zip(keys, choices):
                candidate[key] = [1 - choice, choice]

            if player == 0:
                value = expected_value(candidate, policy)
            else:
                value = -expected_value(policy, candidate)

            best_value = max(best_value, value)

        best_responses.append(best_value)

    game_value = expected_value(policy, policy)
    exploitability = sum(best_responses) / 2
    return game_value, exploitability


def play_against_ai(policy):
    balance = 0
    hand_number = 0

    while True:
        human = hand_number % 2
        cards = random.choice(ALL_DEALS)
        history = ''

        print(f'\nYour card: {CARD_NAMES[cards[human]]}')
        print(f'You are Player {human + 1}')

        while history not in FINISHED_HANDS:
            player = len(history) % 2
            moves = legal_actions(history)

            if player == human:
                while True:
                    options = '/'.join(ACTION_NAMES[action] for action in moves)
                    answer = input(f'{options}: ').strip().lower()
                    matching = [action for action in moves if ACTION_NAMES[action] == answer]
                    if matching:
                        action = matching[0]
                        break
                    print('Please enter one of the listed actions.')
            else:
                probabilities = policy[(cards[player], history)]
                action = random.choices(moves, weights=probabilities)[0]
                print('AI:', ACTION_NAMES[action])

            history += action

        gain = hand_payoff(cards, history)
        if human == 1:
            gain = -gain

        balance += gain
        if not history.endswith('f'):
            print('AI card:', CARD_NAMES[cards[1 - human]])
        print(f'You won/lost: {gain:+d} chips')
        print(f'Your total:   {balance:+d} chips')

        hand_number += 1
        if input('Play again? (y/n): ').strip().lower() != 'y':
            break


def save_results(policy, results):
    with open('strategy.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['card', 'history', 'action', 'probability'])

        for (card, history), probabilities in policy.items():
            for action, probability in zip(legal_actions(history), probabilities):
                writer.writerow([CARD_NAMES[card], history or 'start',
                                 ACTION_NAMES[action], probability])

    with open('training.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'expected_value', 'exploitability'])
        writer.writerows(results)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib if you want to save the training graph.')
        return

    iterations = [row[0] for row in results]
    exploitability = [row[2] for row in results]
    plt.plot(iterations, exploitability)
    plt.xscale('log')
    plt.xlabel('Training iterations')
    plt.ylabel('Exploitability (chips per hand)')
    plt.title('Kuhn Poker CFR Training')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=160)
    plt.close()


def main():
    solver = PokerSolver()
    results = solver.train(10000)
    policy = solver.get_policy()
    save_results(policy, results)

    print('\nTheoretical game value:', -1 / 18)
    print('\nLearned strategy:')

    for (card, history), probabilities in policy.items():
        moves = legal_actions(history)
        descriptions = []
        for action, probability in zip(moves, probabilities):
            descriptions.append(f'{ACTION_NAMES[action]}: {probability:.1%}')
        print(CARD_NAMES[card], history or 'start', ', '.join(descriptions))

    if input('\nPlay against AI? (y/n): ').strip().lower() == 'y':
        play_against_ai(policy)


if __name__ == '__main__':
    main()
