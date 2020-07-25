import random
from collections import namedtuple


class Card(namedtuple('Card', ['rank', 'suit'])):
    def __eq__(self, other):
        return self.rank == other.rank

    def __int__(self):
        if self.rank in ['Jack', 'Queen', 'King']:
            rank = 10
        elif self.rank == 'Ace':
            rank = 1  # Leave `Ace` as 1, for the simplicity of computing state-value
        else:
            rank = int(self.rank)
        return rank


def deck_sum(deck):
    total = 0
    usable_ace = True
    ace_num = 0
    for card in deck:
        if card.rank in ['Jack', 'Queen', 'King']:
            total += 10
        elif card.rank == 'Ace':
            ace_num += 1
        else:
            total += int(card.rank)
    for _ in range(ace_num):
        if total + 11 > 21:
            total += 1
            usable_ace = False
        else:
            total += 11
            usable_ace = True
    return total, usable_ace


class BlackJackEnv(object):
    def __init__(self, dealer_policy, seed=None):
        '''
        NB: An infinite deck is considered, so that there is no advantage
            to keeping track of the cards already dealt.
        '''
        self.dealer_policy = dealer_policy
        if seed:
            print('Set seed to {}!'.format(seed))
            random.seed(seed)
        self.dealer_deck = []
        self.player_deck = []
        self.action_space = [0, 1]
        self.actions = ['hit', 'stick']

        # Init standard 52-card deck
        self._cards = []
        for suit in ['Spade', 'Heart', 'Club', 'Diamond']:
            for rank in [
                    'Ace',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7',
                    '8',
                    '9',
                    '10',
                    'Jack',
                    'Queen',
                    'King',
            ]:
                card = Card(rank=rank, suit=suit)
                self._cards.append(card)
        print('Deck initialized, please call `reset()` to start dealing.')

    def _draw(self, replacement=True):
        return random.choice(self._cards)

    def _is_natural(self, deck):
        cards = [card.rank for card in deck]
        return len(cards) == 2 and 'Ace' in cards and ('Jack' in cards
                                                       or 'Queen' in cards
                                                       or 'King' in cards)

    def shuffle(self):
        random.shuffle(self._cards)
        # print('Deck shuffled, {} cards remained.'.format(len(self._cards)))

    def step(self, act):
        action = self.actions[act]
        done = False
        player_busted = False
        dealer_bustsed = False

        if action == 'hit':
            card = self._draw()
            self.player_deck.append(card)

            # Check go bust
            player_sum, _ = deck_sum(self.player_deck)
            if player_sum > 21:
                player_busted = True
                done = True

        elif action == 'stick':
            # Simulate to dealer's `stick` or busted
            while True:
                act = self.actions[self.dealer_policy(self.dealer_deck)]
                if act == 'hit':
                    card = self._draw()
                    self.dealer_deck.append(card)
                elif act == 'stick':
                    break

                # Check go bust
                dealer_sum, _ = deck_sum(self.dealer_deck)
                if dealer_sum > 21:
                    dealer_bustsed = True
                    break
            done = True

        # Compute reward
        if done:
            if player_busted:
                reward = -1  # lose
            elif dealer_bustsed:
                reward = 1  # win
            else:
                player_sum, _ = deck_sum(self.player_deck)
                dealer_sum, _ = deck_sum(self.dealer_deck)
                if abs(21 - player_sum) < abs(21 - dealer_sum):
                    reward = 1  # win
                elif abs(21 - player_sum) == abs(21 - dealer_sum):
                    reward = 0  # draw
                else:
                    reward = -1  # lose
        else:
            reward = 0

        # Check whether the current desk is natural (in the case, both of the
        # dealer and the player chooses to stick s.t. their policies.)
        if action == 'stick' and self._is_natural(self.player_deck):
            if self._is_natural(self.dealer_deck):
                reward = 0  # draw
            else:
                reward = 1  # win

        return self.numerical_state, reward, done, {}

    def reset(self):
        self.dealer_deck.clear()
        self.player_deck.clear()
        self.shuffle()
        for i in range(2):
            self.player_deck.append(self._draw())
            self.dealer_deck.append(self._draw())
        self.turn = 'player'
        return self.numerical_state

    @property
    def _obs(self):
        return {
            'dealer_showing_card': self.dealer_deck[0],
            'player_deck': self.player_deck
        }

    @property
    def numerical_state(self):
        player_sum, usable_ace = deck_sum(self.player_deck)
        return np.array(
            [player_sum, int(self.dealer_deck[0]),
             int(usable_ace)])
