import datetime
import itertools
import os
import sys
from tkinter import filedialog

from typing import Any

import numpy
import matplotlib.pyplot as plt
import pygame

from Source.constants import DISPLAY_SIZE, SCREEN_SIZE, FPS, FOOD_COLOR, \
    AGENT_COLOR, BACKGROUND_COLOR, AGENTS_AMOUNT, FOOD_AMOUNT
from Source.agent import Agent, Food


class Simulation:
    def __init__(self):
        self._display_ = pygame.Surface(DISPLAY_SIZE)
        self._screen_ = pygame.display.set_mode(SCREEN_SIZE)
        self._clock_ = pygame.time.Clock()

        self._agents_ = self.init_agents(AGENTS_AMOUNT, weights=None)
        self._food_ = self.init_food(FOOD_AMOUNT)
        self._food_coordinates_ = {food.get_coordinates(): food for food in self._food_}
        self._counter_ = 1
        self._population_ = []
        self._food_amount_ = []

    def run(self) -> None:
        while True:
            self.handle_events()
            self.update()

            self.draw()

            self._screen_.blit(pygame.transform.scale(self._display_, SCREEN_SIZE), (0, 0))
            pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.draw_population()

                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self._agents_ = self._agents_[:len(self._agents_) // 2]
                if event.key == pygame.K_SPACE:
                    self._food_coordinates_ = {
                        key: value for key, value in
                        list(self._food_coordinates_.items())[:3 * len(self._food_coordinates_) // 4]
                    }
                if event.key == pygame.K_a:
                    self._agents_ += self.init_agents(AGENTS_AMOUNT // 5)
                if event.key == pygame.K_f:
                    new_food_list = self.init_food(FOOD_AMOUNT // 5)
                    for new_food in new_food_list:
                        self._food_coordinates_[new_food.get_coordinates()] = new_food

                if event.key == pygame.K_s:
                    current_date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    if not os.path.exists(f"Weights/{current_date_string}"):
                        os.mkdir(f"Weights/{current_date_string}")

                    for index, array in enumerate(numpy.random.choice(self._agents_).get_pure_gens()):
                        numpy.save(f"Weights/{current_date_string}/{index}.npy", array)

                if event.key == pygame.K_l:
                    folder_path = filedialog.askdirectory()
                    if folder_path:
                        weights = [numpy.load(os.path.join(folder_path, file_name))
                                   for file_name in os.listdir(folder_path)]
                    else:
                        weights = None

                    for agent in self._agents_:
                        agent.set_weight(weights)

    def update(self):
        self._clock_.tick(FPS)
        self._counter_ += 1

        self._display_.fill(BACKGROUND_COLOR)

        self.move_agents()
        self.update_population()

    def draw(self):
        self.draw_objects(self._agents_)
        self.draw_objects(self._food_coordinates_.values())

    def move_agents(self):
        for agent in self._agents_:
            x, y = agent.get_coordinates()
            possible_food = []

            for i, j in itertools.product(range(-1, 2), range(-1, 2)):
                coordinates = (x - i, y - j)

                if coordinates in self._food_coordinates_:
                    possible_food.append(self._food_coordinates_[coordinates])
                else:
                    continue

            agent.move(agent.decision([food.get_coordinates() for food in possible_food]))
            agent.get_older()

            self.handle_collision(agent, possible_food)

            if agent.check_die():
                self.handle_agent_death(agent)

    def handle_collision(self, agent: Agent, possible_food: list):
        possible_food_coordinates = [food.get_coordinates() for food in possible_food]
        collision_index = agent.get_rect().collidelist([food.get_rect() for food in possible_food])

        if collision_index != -1:
            del self._food_coordinates_[possible_food_coordinates[collision_index]]

            new_food = Food(FOOD_COLOR)
            self._food_coordinates_[new_food.get_coordinates()] = new_food

            agent.eat()

    def handle_agent_death(self, agent):
        if agent.get_check_offspring():
            for _ in range(agent.get_eat_counter()):
                r, g, b = agent.get_color()
                close_to_parent_color = (numpy.random.randint(max(r - 25, 0), min(r + 25, 255)),
                                         numpy.random.randint(max(g - 25, 0), min(g + 25, 255)),
                                         numpy.random.randint(max(b - 25, 0), min(b + 25, 255)))

                self._agents_.append(Agent(agent.get_gens_with_mutation(), close_to_parent_color))

        self._agents_.remove(agent)

    def update_population(self):
        self._population_.append(len(self._agents_))
        self._food_amount_.append(len(self._food_coordinates_))

    @staticmethod
    def init_agents(amount: int, weights=None) -> list:
        agents = []
        for _ in range(amount):
            agents.append(Agent(weights=weights, color=AGENT_COLOR))
        return agents

    @staticmethod
    def init_food(amount: int) -> list:
        return [Food(FOOD_COLOR) for _ in range(amount)]

    def draw_objects(self, objects: list | numpy.ndarray | Any) -> None:
        for simulation_object in objects:
            pygame.draw.rect(self._display_, simulation_object.get_color(), simulation_object.get_rect())

    def draw_population(self) -> None:
        plt.plot(range(len(self._population_)), self._population_)
        plt.plot(range(len(self._food_amount_)), self._food_amount_)
        plt.show()
