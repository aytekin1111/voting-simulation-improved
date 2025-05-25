from collections import namedtuple
from typing import List, Tuple, Optional

import pygame
import pygame.gfxdraw

import numpy as np

# Initialize pygame
pygame.init()
pygame.mixer.init()  # Initialize sound mixer

# Candidate attributes
Candidate = namedtuple("Candidate", "name color initial_percentage")


class Voter:
    """A voter is a cellular automatum that has a memory of the candidates and their scores.
        It can influence and get influenced by its neighbors.

    Attributes:
        candidate_memory (np.ndarray): A numpy array that stores the voter's feelings towards the candidates
        influence_factor (np.float32): The factor that determines how much influence the voter can produce
        conceding_factor (np.float32): The factor that determines how much influence the voter can concede
        override_color (Optional[Tuple[int, int, int]]): The color that the voter will be drawn with
        override_color_timer (int): The timer that determines how long the override color will be active
        memory_display (bool): Whether the voter's memory will be displayed or not
    """

    def __init__(self):

        # Main functionallity
        self.candidate_memory: np.ndarray = np.empty(0)
        self.influence_factor : np.float32 = np.random.rand() * VOTER_PRODUCE_INFLUENCE_FACTOR
        self.conceding_factor : np.float32 = np.random.rand() * VOTER_CONCEDE_INFLUENCE_FACTOR

        # For visualization purposes
        self.override_color : Optional[Tuple[int, int, int]] = None
        self.override_color_timer : int = 0

        self.memory_display : bool = False

    # Simulation functions

    def initialize_candidate_memory(
        self, candidates: dict, ordered_candidates: List[str]
    ):
        """Initializes the candidate memory of the voter

        Args:
            candidates (dict): A dictionary that maps candidate names to their Candidate objects
            ordered_candidates (List[str]): A list of candidate names in the order that they will be displayed

        Notes:
            The candidate memory is initialized by generating random scores for the candidates and then reordering them
            based on the initial percentages of the candidates. This is done to simulate the fact that voters are more
            likely to vote for candidates that have higher initial percentages.
        """
        # Generate random scores to assign to the candidates
        scores = np.random.randn(len(candidates))

        # Sort in descending order
        scores = sorted(scores, reverse=True)

        # Create a dictionary that maps candidate names to their initial percentages
        candidate_percentages = {
            i: candidates[candidate].initial_percentage
            for i, candidate in enumerate(ordered_candidates)
        }

        # Create the of candidate names in a random permutation as described in the docstring
        candidate_permutation = np.random.choice(
            len(candidate_percentages),
            len(candidate_percentages),
            replace=False,
            p=[candidate_percentages[candidate] for candidate in candidate_percentages],
        )

        # Reorder the scores array based on the candidate_permutation
        reordered_scores = np.zeros_like(scores)
        reordered_scores[candidate_permutation] = scores

        # Assign the reordered scores to the candidate memory
        self.candidate_memory = reordered_scores

    def memory_normalize(self):
        """Applies softmax to the candidate memory"""
        self.candidate_memory = normalize(self.candidate_memory)

    def vote(self, ordered_candidates: List[str]):
        """Returns the candidate that the voter will most likely vote for"""
        return ordered_candidates[np.argmax(self.candidate_memory)]

    def produce_influence(self):
        """Produces influence for the candidates in the candidate memory
            The influence is determined by the candidate's score in the candidate memory and the influence factor

        Returns:
            influence (dict): A dictionary that maps candidate names to their produced influence
        """

        # Produce influence for each candidate using the voter's feelings towards the candidate
        # The gaussian noise is added in such a way that 3 standard deviations is 0.1 and the candidate memory values
        influence = (
            np.random.randn() * 0.033 + self.candidate_memory
        ) * self.influence_factor

        return influence

    def concede_influence(self, influence, factor=1.0):
        """Concedes influence to the candidates in the candidate memory"""
        self.candidate_memory += influence * self.conceding_factor * factor

    def color_override(self, color, duration):
        """Overrides the color of the voter for a certain duration"""
        self.override_color = color
        self.override_color_timer = duration

    def update(self):
        """Updates the voter"""
        # Update the override color timer
        if self.override_color_timer > 0:
            self.override_color_timer -= 1
        else:
            self.override_color = None

    # Visualization functions

    def draw_memory(self, screen, x, y, candidates, ordered_candidates, font):
        """Draws the candidate memory of the voter"""
        # Draw the background
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (x + VOTER_RADIUS + 5, y - 10, 140, 20 * len(candidates)),
        )

        # Draw the candidate memory
        for i, candidate in enumerate(ordered_candidates):
            # Draw the candidate name
            text = font.render(candidate, True, (0, 0, 0))
            screen.blit(text, (x + VOTER_RADIUS + 10, y - 10 + 20 * i))

            # Draw the candidate score
            text = font.render(
                str(round(self.candidate_memory[i], 2)), True, (0, 0, 0)
            )
            screen.blit(text, (x + VOTER_RADIUS + 100, y - 10 + 20 * i))

    def draw(self, screen, x, y, candidates, ordered_candidates, font):
        """Draws the voter"""
        if self.override_color:
            pygame.draw.circle(
                screen, self.override_color, (int(x), int(y)), VOTER_RADIUS
            )
        else:
            pygame.draw.circle(
                screen,
                candidates[self.vote(ordered_candidates)].color,
                (int(x), int(y)),
                VOTER_RADIUS,
            )

        if self.memory_display:
            self.draw_memory(screen, x, y, candidates, ordered_candidates, font)

    def handle_event(self, event, x, y):
        """When hovering over a voter, the voter's candidate memory is displayed"""
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            if (
                x - VOTER_RADIUS <= mouse_pos[0] <= x + VOTER_RADIUS
                and y - VOTER_RADIUS <= mouse_pos[1] <= y + VOTER_RADIUS
            ):
                self.memory_display = True
            else:
                self.memory_display = False


class Slider:
    """A slider that can be used to change the value of a variable"""

    def __init__(self, x, y, w, h, min_value, max_value, value, color, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.color = color
        self.label = label

        self.dragging = False

    def handle_event(self, event):
        """Handles events"""

        # If the mouse is pressed down and the mouse is within the slider, start dragging
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if (
                    self.x <= event.pos[0] <= self.x + self.w
                    and self.y <= event.pos[1] <= self.y + self.h
                ):
                    self.dragging = True

        # If the mouse is released, stop dragging
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        # If the mouse is moved and the slider is being dragged, update the value
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                if event.pos[0] < self.x:
                    self.value = self.min_value
                elif event.pos[0] > self.x + self.w:
                    self.value = self.max_value
                else:
                    self.value = (
                        self.min_value
                        + (self.max_value - self.min_value)
                        * (event.pos[0] - self.x)
                        / self.w
                    )

    def draw(self, screen, font):
        """Draws the slider"""

        # Draw the bar
        pygame.draw.rect(
            screen, self.color, (self.x, self.y, self.w, self.h), border_radius=2
        )

        # Draw the value indicator
        pygame.draw.rect(
            screen,
            (30, 30, 30),
            (
                self.x
                + (self.value - self.min_value)
                / (self.max_value - self.min_value)
                * self.w
                - 5,
                self.y - 2.5,
                10,
                self.h + 5,
            ),
            border_radius=5,
        )

        # Draw the label on top of the bar
        text = font.render(self.label, True, BLACK)
        screen.blit(
            text,
            (
                self.x + self.w / 2 - text.get_width() / 2,
                self.y - text.get_height() - 5,
            ),
        )


class Button:
    """A button that can be used to trigger an action"""

    def __init__(self, x, y, w, h, color, text_emoji):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text_emoji = text_emoji

        # Create a short click animation
        self.click_animation = 0

    def handle_event(self, event):
        """Handles events"""

        # If the mouse is pressed down and the mouse is within the button, trigger the action
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if (
                    self.x <= event.pos[0] <= self.x + self.w
                    and self.y <= event.pos[1] <= self.y + self.h
                ):
                    # Also trigger a short click animation
                    self.click_animation = 5
                    return True
        
        return False

    def update(self):
        """Updates the button"""
        # Update the click animation
        if self.click_animation > 0:
            self.click_animation -= 1

    def draw(self, screen, font):
        """Draws the button"""

        # Draw the button
        if self.click_animation > 0:
            # If the button is being clicked, draw a darker version of the button
            pygame.draw.rect(
                screen,
                (max(self.color[0] - 50, 0), max(self.color[1] - 50, 0), max(self.color[2] - 50, 0)),
                (self.x, self.y, self.w, self.h),
                border_radius=2,
            )
        else:
            pygame.draw.rect(
                screen, self.color, (self.x, self.y, self.w, self.h), border_radius=2
            )

        # Draw the text emoji on the button
        text = font.render(self.text_emoji, True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.x + self.w/2, self.y + self.h/2))
        screen.blit(text, text_rect)


class Propaganda:
    """A propaganda that can be used to influence voters.
    The infuence radius starts at 0 and increases over time until it reaches a random radius.
    Voters that are within the influence radius will be influenced by the propaganda.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        candidate: str,
        ordered_candidates: List[str],
        is_negative: bool = False,
    ):

        self.is_negative = is_negative

        # When a propaganda is triggered, it will affect the voters about a candidate
        self.label_idx = ordered_candidates.index(candidate)
        self.label = ordered_candidates[self.label_idx]

        # A propaganda is triggered at a random location with a random radius
        # Here the radius is a parameter that can be played with
        self.influence_center = np.random.rand(2)
        self.influence_center[0] = self.influence_center[0] * SCREEN_WIDTH
        self.influence_center[1] = (
            (self.influence_center[1] * (SCREEN_HEIGHT - INFO_HEIGHT - INFO_BOX_MARGIN))
            + INFO_HEIGHT
            + INFO_BOX_MARGIN
        )
        self.influence_radius = (
            np.random.rand() * (MAX_PROPAGANDA_RADIUS - MIN_PROPAGANDA_RADIUS)
            + MIN_PROPAGANDA_RADIUS
        )

        # The current radius of the propaganda is initialized to 0
        self.current_radius = 0

        # Calculate the distance between the influence center and all the voters
        self.influence_center_dists = np.linalg.norm(
            coordinates - self.influence_center, axis=1
        )

        # Influenced voters will be updated when the current radius of the propaganda is increased
        self.influenced_voters = set()

        # Create influence vector of the propaganda

        # magnitude of the influence
        self.influence = PROPAGANDA_STRENGTH
        if self.is_negative:
            self.influence = -self.influence

        # Create propaganda vector
        self.propaganda_influence = np.zeros(len(ordered_candidates))
        self.propaganda_influence[self.label_idx] = self.influence

    def update(self, voters: List[Voter]):
        """Updates the influence area of the propaganda
        and updates the influenced voters."""

        # If the distance between the influence center and the voter is less than
        # the current radius, the voter is influenced
        influenced_voter_map = self.influence_center_dists < self.current_radius
        # Get the indices of the influenced voters
        influenced_voters = set(np.where(influenced_voter_map)[0])

        remaining_time = (
            self.influence_radius - self.current_radius
        ) / PROPAGANDA_D_RADIUS

        # Update the influenced voters
        for voter in influenced_voters:
            voters[voter].color_override(PROPAGANDA_OVERRIDE_COLOR, remaining_time)
            voters[voter].concede_influence(self.propaganda_influence)

        # Update the set of influenced voters
        self.influenced_voters = influenced_voters

        # Increase the radius of the propaganda
        self.current_radius += PROPAGANDA_D_RADIUS

    def draw(self, screen, candidates: dict[str, Candidate], font):
        """Draws the propaganda on the screen"""
        pygame.gfxdraw.aacircle(
            screen,
            int(self.influence_center[0]),
            int(self.influence_center[1]),
            int(self.current_radius),
            candidates[self.label].color,
        )
        pygame.gfxdraw.aacircle(
            screen,
            int(self.influence_center[0]),
            int(self.influence_center[1]),
            max(0, int(self.current_radius) - 1),
            candidates[self.label].color,
        )

        # Draw text emoji in the center
        emoji_text = ">:->" if self.is_negative else "0:-)"
        text = font.render(emoji_text, True, (0, 0, 0))
        text_rect = text.get_rect(center=(int(self.influence_center[0]), int(self.influence_center[1])))
        screen.blit(text, text_rect)

    def is_finished(self):
        """Returns True if the propaganda is finished"""
        return self.current_radius > self.influence_radius


class PropagandaButtonTrigger:
    """Consists of a label for candidate and two buttons,
    one for positive propaganda and one for negative propaganda.
    The label will be on the top and the two buttons will be on the buttom side
    by side.
    """

    def __init__(
        self,
        x,
        y,
        button_w,
        button_h,
        color,
        label,
    ):

        self.x = x
        self.y = y
        self.button_w = button_w
        self.button_h = button_h
        self.color = color
        self.label = label

        self.positive_button = Button(
            self.x + BUTTON_HORIZONTAL_MARGIN,
            self.y + BUTTON_VERTICAL_MARGIN,
            self.button_w,
            self.button_h,
            self.color,
            "0:-)"
        )

        self.negative_button = Button(
            self.x - self.button_w - BUTTON_HORIZONTAL_MARGIN,
            self.y + BUTTON_VERTICAL_MARGIN,
            self.button_w,
            self.button_h,
            self.color,
            ">:->"
        )

    def handle_event(self, event, propaganda_queue: List[Propaganda], coordinates: np.ndarray, ordered_candidates: List[str]):
        """Handles events"""

        if self.positive_button.handle_event(event):
            self.trigger_positive_propaganda(
                propaganda_queue, coordinates, ordered_candidates
            )
        elif self.negative_button.handle_event(event):
            self.trigger_negative_propaganda(
                propaganda_queue, coordinates, ordered_candidates
            )

    def draw(self, screen, font):

        # Draw the buttons
        self.positive_button.draw(screen, font)
        self.negative_button.draw(screen, font)

        # Draw the label on top of the buttons centered
        label = font.render(self.label, True, (0, 0, 0))
        label_rect = label.get_rect()
        label_rect.center = (self.x, self.y)
        screen.blit(label, label_rect)

    def update(self):
        """Updates the buttons"""
        self.positive_button.update()
        self.negative_button.update()

    def trigger_positive_propaganda(
        self,
        propaganda_queue: List[Propaganda],
        coordinates: np.ndarray,
        ordered_candidates: List[str],
    ):
        """Triggers a positive propaganda"""

        # If the propaganda is triggered, add it to the list of propagandas
        propaganda_queue.append(
            Propaganda(coordinates, self.label, ordered_candidates, is_negative=False)
        )

    def trigger_negative_propaganda(
        self,
        propaganda_queue: List[Propaganda],
        coordinates: np.ndarray,
        ordered_candidates: List[str],
    ):
        """Triggers a negative propaganda"""

        # If the propaganda is triggered, add it to the list of propagandas
        propaganda_queue.append(
            Propaganda(coordinates, self.label, ordered_candidates, is_negative=True)
        )


class Simulation:
    """A simulation of a voting system. This class is used to handle
    updates and draws of the simulation.
    """

    def __init__(
        self,
        candidates,
        num_voters,
        k_neighbours,
        screen_width,
        screen_height,
        info_height,
        info_box_margin,
        propaganda_prob,
    ):

        self.candidates = candidates
        self.num_voters = num_voters
        self.k_neighbours = k_neighbours
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.info_height = info_height
        self.info_box_margin = info_box_margin
        self.propaganda_prob = propaganda_prob

        self.running = False

        # Election timer (3 minutes = 180 seconds) - update every frame for smooth countdown
        self.election_time_limit = 180 * FPS  # Convert to frames
        self.election_timer = self.election_time_limit
        self.election_finished = False
        
        # For smooth timer display, track last displayed second
        self.last_displayed_second = 180
        
        # Sound effects - DISABLED for now
        self.crowd_sound_playing = False
        # self.create_crowd_sound()  # Commented out to disable sound

        # Set up the drawing window
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])

        # Set title
        pygame.display.set_caption("Voting Simulation")

        # Set up the clock
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 20, bold=True)

        # Frame counter
        self.frame = 0

        # Ordered candidates
        self.ordered_candidates = sorted(list(self.candidates.keys()))

        # Create voters a list of Voter objects
        self.voters: List[Voter] = [Voter() for _ in range(num_voters)]

        # Create uniformly distributed coordinates for each voter
        self.coordinates = np.random.rand(self.num_voters, 2)
        self.coordinates[:, 0] = self.coordinates[:, 0] * self.screen_width
        self.coordinates[:, 1] = self.coordinates[:, 1] * (self.screen_height - self.info_height)

        # Initialize speed vectors for each voter
        self.speed_vectors = np.zeros((self.num_voters, 2))

        # Initialize candidate memory for each voter
        for voter in self.voters:
            voter.initialize_candidate_memory(candidates, self.ordered_candidates)

        self.propaganda_queue: List[Propaganda] = []

        # Create propaganda button triggers
        self.propaganda_button_triggers: List[PropagandaButtonTrigger] = []
        trigger_seperation = 60  # Reduced spacing
        trigger_top_margin = 30  # Fixed top margin instead of calculated
        # Move buttons more to the left to prevent text overflow
        button_x_position = self.screen_width - 150  # More space for text
        for i, candidate in enumerate(self.candidates.keys()):
            self.propaganda_button_triggers.append(
                PropagandaButtonTrigger(
                    button_x_position,
                    trigger_top_margin + i * trigger_seperation,
                    35,  # Smaller button width
                    35,  # Smaller button height
                    self.candidates[candidate].color,
                    candidate
                )
            )


    def _fill_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        theta0: float,
        theta1: float,
        color: Tuple[int, int, int],
        ndiv: int = 50,
    ):
        """Draws a filled arc using aapolygon"""

        x0, y0 = center

        dtheta = (theta1 - theta0) / ndiv
        angles = [theta0 + i * dtheta for i in range(ndiv + 1)]

        points = [(x0, y0)] + [
            (x0 + radius * np.cos(theta), y0 - radius * np.sin(theta))
            for theta in angles
        ]

        # Turn into a list of integers
        points = [(int(x), int(y)) for x, y in points]

        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def draw_pie_chart(self, x: int, y: int, radius: int, votes: dict):
        """Draws a pie chart on the screen

        Args:
            x (int): center x coordinate of the pie chart
            y (int): center y coordinate of the pie chart
            radius (int): radius of the pie chart
            votes (dict): dictionary of votes
        """
        # Sum of all votes in the dictionary to calculate the percentage
        total = sum(votes.values())

        label_centers = []

        # Draw the pie chart
        start_angle = 0
        for candidate in votes:

            # Calculate the percentage of votes for the candidate
            percentage = votes[candidate] / total

            # Calculate the end angle of the arc
            end_angle = start_angle + percentage * 2 * np.pi

            # Draw the arc
            self._fill_arc(
                (x, y), radius, start_angle, end_angle, self.candidates[candidate].color
            )

            # Draw label and percentage on the pie chart
            if percentage > 0.02:

                pie_center = (
                    x + radius * np.cos(start_angle + (end_angle - start_angle) / 2),
                    y - radius * np.sin(start_angle + (end_angle - start_angle) / 2),
                )

                label_centers.append((f"{candidate}: {percentage * 100:.2f}%", pie_center))

            start_angle = end_angle

        # Draw the labels
        for label, center in label_centers:
            text = self.font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)


    def random_walk(self):
        """Performs a random walk on the coordinates of the voters"""

        random_walk = np.random.randn(*self.coordinates.shape) * VOTER_MOVE_FACTOR

        self.speed_vectors = random_walk * 0.9 + self.speed_vectors * 0.1

        self.coordinates += self.speed_vectors #np.random.randn(*self.coordinates.shape) * 0.2

        # Loop the frame 
        self.coordinates[:, 0][self.coordinates[:, 0] < 0] += self.screen_width
        self.coordinates[:, 0][self.coordinates[:, 0] > self.screen_width] -= self.screen_width
        self.coordinates[:, 1][self.coordinates[:, 1] < 0] += self.screen_height - self.info_height
        self.coordinates[:, 1][self.coordinates[:, 1] > self.screen_height] -= self.screen_height - self.info_height

    def update(self):
        """Updates the simulation"""
        
        # Update election timer - decrease every frame
        if not self.election_finished:
            self.election_timer -= 2  # Decrease by 2 to make it faster since we reduced FPS to 30
            
            if self.election_timer <= 0:
                self.election_finished = True
                # Stop crowd sound when election ends - DISABLED
                # self.stop_crowd_sound()
                print("SECIM BITTI! Final Sonuclar:")
                votes = {candidate: 0 for candidate in self.candidates}
                for voter in self.voters:
                    votes[voter.vote(self.ordered_candidates)] += 1
                for candidate, vote_count in votes.items():
                    percentage = (vote_count / len(self.voters)) * 100
                    print(f"   {candidate}: {vote_count} oy (%{percentage:.1f})")

        # Update the propaganda queue
        self.propaganda_queue = [
            propaganda
            for propaganda in self.propaganda_queue
            if not propaganda.is_finished()
        ]

        # Only continue simulation if election is not finished
        if not self.election_finished:
            # Perform random walks on the coordinates of the voters
            self.random_walk()

            # Calculate the distances between each voter
            distances = np.linalg.norm(
                self.coordinates[:, None, :] - self.coordinates[None, :, :], axis=2
            )

            # Calculate the influence of each voter
            influences = [voter.produce_influence() for voter in self.voters]

            # Update the voters
            for i, voter in enumerate(self.voters):
                # Find the indices of the K nearest neighbors
                nearest_neighbors = np.argsort(distances[i])[1 : self.k_neighbours + 1]

                # Update the voter
                for neighbor in nearest_neighbors:
                    voter.concede_influence(
                        influences[neighbor], factor=1 / self.k_neighbours
                    )
                voter.update()

            # Update the propagandas
            for propaganda in self.propaganda_queue:
                propaganda.update(self.voters)

            # Normalize the candidate memory of each voter
            for voter in self.voters:
                voter.memory_normalize()

        for propaganda_button_trigger in self.propaganda_button_triggers:
            propaganda_button_trigger.update()
    
    def create_crowd_sound(self):
        """Creates procedural crowd noise using numpy and pygame"""
        try:
            # Create white noise for crowd base using numpy
            import numpy as np
            
            sample_rate = 22050
            duration = 3.0  # 3 seconds of crowd noise
            frames = int(duration * sample_rate)
            
            # Generate crowd noise
            # Mix of different frequencies to simulate crowd
            time = np.linspace(0, duration, frames)
            
            # Base crowd rumble (low frequency)
            crowd_base = np.sin(2 * np.pi * 120 * time) * 0.2
            
            # Add some random noise for chatter
            noise = np.random.normal(0, 0.1, frames)
            
            # Add some periodic sounds for applause/reactions
            applause = np.sin(2 * np.pi * 400 * time) * 0.1 * (np.sin(2 * np.pi * 0.5 * time) + 1)
            
            # Combine all sounds
            crowd_sound = crowd_base + noise + applause
            
            # Normalize to prevent clipping
            crowd_sound = crowd_sound / np.max(np.abs(crowd_sound)) * 0.3
            
            # Convert to 16-bit integers
            crowd_sound_int = (crowd_sound * 32767).astype(np.int16)
            
            # Create stereo sound (duplicate mono to stereo) - FIX C-contiguous issue
            stereo_sound = np.column_stack((crowd_sound_int, crowd_sound_int))
            
            # Ensure the array is C-contiguous
            stereo_sound = np.ascontiguousarray(stereo_sound)
            
            # Create pygame sound
            self.crowd_sound = pygame.sndarray.make_sound(stereo_sound)
            
            print("Kalabali sesi olusturuldu!")
            
        except Exception as e:
            print(f"Ses olusturulamadi: {e}")
            print("Daha basit ses deneniyor...")
            
            # Fallback: Try simple beep sound
            try:
                # Create simple sine wave
                import numpy as np
                sample_rate = 22050
                duration = 1.0
                frequency = 200
                frames = int(duration * sample_rate)
                
                sine_wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames)) * 0.1
                sine_wave_int = (sine_wave * 32767).astype(np.int16)
                
                # Simple mono to stereo conversion
                simple_stereo = np.zeros((frames, 2), dtype=np.int16)
                simple_stereo[:, 0] = sine_wave_int
                simple_stereo[:, 1] = sine_wave_int
                
                self.crowd_sound = pygame.sndarray.make_sound(simple_stereo)
                print("Basit ses olusturuldu!")
                
            except Exception as e2:
                print(f"Basit ses de olusturulamadi: {e2}")
                self.crowd_sound = None
    
    def play_crowd_sound(self):
        """Plays crowd sound if available"""
        if hasattr(self, 'crowd_sound') and self.crowd_sound and not self.crowd_sound_playing:
            try:
                self.crowd_sound.play(loops=-1, fade_ms=500)  # Loop indefinitely with fade in
                self.crowd_sound_playing = True
                print("Kalabali sesi basladi!")
            except:
                pass
    
    def stop_crowd_sound(self):
        """Stops crowd sound"""
        if hasattr(self, 'crowd_sound') and self.crowd_sound and self.crowd_sound_playing:
            try:
                self.crowd_sound.fadeout(1000)  # Fade out over 1 second
                self.crowd_sound_playing = False
                print("Kalabali sesi durdu!")
            except:
                pass

    def draw(self):
        """Draws the simulation"""

        # Fill the background with white
        self.screen.fill(BACKGROUND_COLOR2)

        # Draw the voters
        for i, voter in enumerate(self.voters):
            voter.draw(
                self.screen,
                self.coordinates[i][0],
                self.coordinates[i][1] + self.info_height,
                self.candidates,
                self.ordered_candidates,
                self.font
            )

        for propaganda in self.propaganda_queue:
            propaganda.draw(self.screen, self.candidates, self.font)

        # Set white background for info box
        pygame.draw.rect(
            self.screen, BACKGROUND_COLOR, (0, 0, self.screen_width, self.info_height)
        )

        # Calculate the votes
        votes = {candidate: 0 for candidate in self.candidates}
        for voter in self.voters:
            votes[voter.vote(self.ordered_candidates)] += 1

        # Draw the pie chart
        self.draw_pie_chart(
            self.info_height // 2 + 30,
            self.info_height // 2,
            self.info_height // 2 - 40,
            votes,
        )

        # Draw the info line
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, self.info_height),
            (self.screen_width, self.info_height),
        )

        # Draw propaganda button triggers (only if election not finished)
        if not self.election_finished:
            for propaganda_button_trigger in self.propaganda_button_triggers:
                propaganda_button_trigger.draw(self.screen, self.font)

        # Draw election timer
        remaining_seconds = max(0, self.election_timer // FPS)
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        if self.election_finished:
            timer_text = "SEÇİM BİTTİ!"
            timer_color = (255, 0, 0)  # Red
        else:
            timer_text = f"Kalan Süre: {minutes:02d}:{seconds:02d}"
            timer_color = (0, 0, 0)  # Black
        
        timer_surface = self.font.render(timer_text, True, timer_color)
        self.screen.blit(timer_surface, (self.screen_width - 200, 10))

    def handle_events(self):
        """Handles the events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Only handle propaganda buttons if election is not finished
            if not self.election_finished:
                for propaganda_button_trigger in self.propaganda_button_triggers:
                    propaganda_button_trigger.handle_event(event, self.propaganda_queue, self.coordinates, self.ordered_candidates)

            for i, voter in enumerate(self.voters):
                voter.handle_event(event, self.coordinates[i][0], self.coordinates[i][1] + self.info_height)

    def run(self):

        self.running = True
        
        # Start crowd sound when simulation begins - DISABLED
        # self.play_crowd_sound()

        while self.running:

            self.handle_events()

            self.update()

            self.draw()

            pygame.display.flip()

            self.frame += 1

            self.clock.tick(FPS)
        
        # Stop crowd sound when simulation ends - DISABLED
        # self.stop_crowd_sound()


def normalize(v):
    """Normalizes a vector"""
    norm = np.sum(np.abs(v))
    if norm == 0:
        return v
    return v / norm


if __name__ == "__main__":

    # Set screen size (optimized for laptop screens)
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 650
    INFO_HEIGHT = 180
    INFO_BOX_MARGIN = 40

    # Create candidates
    CANDIDATES = {
        "Recep": Candidate("Recep", (245, 126, 32), 0.4),
        "Muharrem": Candidate("Muharrem", (13, 93, 166), 0.1),
        "Sinan": Candidate("Sinan", (192, 13, 14), 0.1),
        "Kemal": Candidate("Kemal", (228, 1, 11), 0.3),
        "Diger": Candidate("Diger", (186, 199, 193), 0.1),
    }

    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    BACKGROUND_COLOR = (80, 80, 80)
    BACKGROUND_COLOR2 = (240, 240, 240)
    FPS = 30  # Reduced FPS to make timer update more visible

    NUM_VOTERS = 750
    VOTER_RADIUS = 3
    K_NEIGHBORS = 5

    BUTTON_HORIZONTAL_MARGIN = 20
    BUTTON_VERTICAL_MARGIN = 15

    MIN_PROPAGANDA_RADIUS = 50
    MAX_PROPAGANDA_RADIUS = 100
    RANDOM_WALK_SEQUENCE_LENGTH = 64
    WALK_QUANTIZATION = 64
    MOMENTUM = 0.9
    VOTER_MOVE_FACTOR = 1.0
    VOTER_PRODUCE_INFLUENCE_FACTOR = 0.05
    VOTER_CONCEDE_INFLUENCE_FACTOR = 0.05

    PROPAGANDA_PROBABILITY = 1.0
    PROPAGANDA_STRENGTH = 1.0
    PROPAGANDA_OVERRIDE_COLOR = None  # (191, 0, 191)
    PROPAGANDA_D_RADIUS = 2

    simulation = Simulation(
        CANDIDATES,
        NUM_VOTERS,
        K_NEIGHBORS,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        INFO_HEIGHT,
        INFO_BOX_MARGIN,
        PROPAGANDA_PROBABILITY,
    )

    simulation.run()

    # Done! Time to quit.
    pygame.quit()