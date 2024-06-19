import numpy as np

class SimpleTasks:

    @staticmethod
    def fixed_interval(interval, iti_halflife=5, iti_min = 3, session_length=30, n_dim = 2, n_sessions = 2, seed = 0, states = False):
        """
        Targets as replicas of one-hot inputs delayed by a fixed interval

        Parameters
        ----------
        interval : int
            Number of steps between input and target
        iti_halflife : int
            Half life of the ITI exponential distribution in steps
        iti_min : int
            Minimum ITI in steps
        session_length : int
            Number of steps in a session
        n_dim : int
            Number of dimensions in the input and output
        n_sessions : int
            Number of sessions to generate
        seed : int
            Random seed

        Returns
        -------
        inputs : array (n_sessions, session_length, n_dim)
            Inputs for each session
        targets : array (n_sessions, session_length, n_dim)
            Targets for each session
        states : array (n_sessions, session_length, 3 + iti_min)
            One-hot state vectors for each session. The the last dimension index
            states as
            - 0: ITI exponential countdown
            - 1 to iti_min: ITI deterministic countdown
            For i in range n_dim
            - (iti_min) + (i * (interval + 1)): Input i
            - (iti_min) + (i * (interval + 1)) + (1 to interval): Interval
            countdown i
            - (iti_min) + (i * (interval + 1)) + (interval + 1): Reward i
            
            
        """
        inputs = np.zeros((n_sessions, session_length, n_dim))
        targets = np.zeros((n_sessions, session_length, n_dim))
        if states:
            states_ = np.zeros((n_sessions, session_length), dtype = int)
        rng = np.random.default_rng(seed)

        for i in range(n_sessions):
            x = 0
            while True:
                # select timing and dimension for next pulse
                iti = rng.exponential(iti_halflife / np.log(2))
                next_x = x + int(np.ceil(iti)) + iti_min - 1
                if next_x + interval >= session_length:
                    break
                ix = rng.integers(n_dim)
                # set input and target for pulse add states for the trial
                inputs[i, next_x, ix] = 1
                targets[i, next_x + interval, ix] = 1
                if states:
                    ix_ofs = iti_min + ix * (interval + 1)
                    states_[i, x + 1 : x + iti_min] = np.arange(iti_min-1) + 1
                    states_[i, x + iti_min : next_x] = 0
                    states_[i, next_x] = ix_ofs
                    states_[i, next_x : next_x + interval] = np.arange(interval) + ix_ofs
                    states_[i, next_x + interval] = ix_ofs + interval
                x = next_x + interval
        
        # convert integer states to one-hot and return
        if states:
            nstates = iti_min + n_dim * (interval + 2)
            states = np.zeros((n_sessions, session_length, nstates))
            for i in range(n_sessions):
                states[i, np.arange(session_length), states_[i]] = 1
            return inputs, targets, states
        return inputs, targets
        
    @staticmethod
    def cdfi(interval, flip_halflife = 12, flip_min = 5, iti_halflife = 5, iti_min = 1, session_length = 30, n_dim = 2, n_sessions = 2, seed = 0):
        """
        Fixed interval task with a context switch
        """
        inp, tgt = SimpleTasks.fixed_interval(interval, iti_halflife, iti_min, session_length, n_dim, n_sessions, seed)
        rng = np.random.default_rng(seed + 1)
        context = np.zeros((n_sessions, session_length, n_dim))
        for i in range(n_sessions):
            x = 0
            ix = rng.integers(n_dim)
            while True:
                flip = rng.exponential(flip_halflife / np.log(2))
                next_x = x + int(np.ceil(flip)) + flip_min - 1
                new_ix = rng.integers(n_dim - 1)
                ix = new_ix if new_ix < ix else new_ix + 1
                if next_x >= session_length:
                    context[i, x:, ix] = 1
                    break
                context[i, x : next_x, ix] = 1
                x = next_x
        tgt = tgt * context
        inp = np.concatenate([inp, context], axis = -1)
        return inp, tgt
    
    @staticmethod
    def ohflip(halflife, tmin, ndim = 2, session_length = 30, n_sessions = 2, seed = 0):
        """
        One-hot flip task.

        With Poisson-timed events of one-hot inputs, maintain activity in the
        corresponding output dimension until the next event.

        Parameters
        ----------
        interval : int
            Number of steps between input and target
        halflife : int
            Half life the exponential defining event timing above `tmin`.
        tmin : int
            Minimum time between events.
        ndim : int
            Number of dimensions in the input and output
        session_length : int
            Number of steps in a session
        n_sessions : int
            Number of sessions to generate
        seed : int
            Random seed

        Returns
        -------
        inputs : array (n_sessions, session_length, ndim)
            Inputs for each session
        targets : array (n_sessions, session_length, ndim)
            Targets for each session
        """
        inp = np.zeros((n_sessions, session_length, ndim))
        tgt = np.zeros((n_sessions, session_length, ndim))
        rng = np.random.default_rng(seed)

        ### generate events, setting a random index of inp to 1 at each event
        #   and setting the corresponding index of tgt to 1 until the next event
        for i in range(n_sessions):
            x = 0
            ix = rng.integers(ndim)
            while True:
                iti = rng.exponential(halflife / np.log(2))
                x += int(np.ceil(iti)) + tmin - 1
                if x >= session_length:
                    break
                new_ix = rng.integers(ndim - 1)
                ix = new_ix if new_ix < ix else new_ix + 1
                inp[i, x, ix] = 1
                tgt[i, x:, :] = 0
                tgt[i, x:, ix] = 1
        
        return inp, tgt