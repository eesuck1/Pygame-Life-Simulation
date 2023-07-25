import numpy
import pygame

from Source.constants import DISPLAY_SIZE, FPS, OBSERVE, AGENT_SIZE, DIRECTIONS
from Source.ML import leaky_relu, softmax


class Agent:
    def __init__(self, weights: numpy.ndarray = None, color: tuple[int, int, int] = (255, 255, 255)):
        if weights is None:
            first_layer = numpy.random.random_sample((24, 24))
            second_layer = numpy.random.random_sample((24, 4))

            weights = [first_layer, second_layer]
        self._weights_ = weights
        self._color_ = color

        self._rect_ = pygame.rect.Rect(*numpy.random.randint(0, DISPLAY_SIZE), AGENT_SIZE, AGENT_SIZE)

        self._health_ = FPS * 2

        self._moves_ = {
            DIRECTIONS[0]: lambda: self.change_coordinates(0, -1),
            DIRECTIONS[1]: lambda: self.change_coordinates(1, 0),
            DIRECTIONS[2]: lambda: self.change_coordinates(0, 1),
            DIRECTIONS[3]: lambda: self.change_coordinates(-1, 0),
        }

        self._can_give_offspring_ = False
        self._eat_counter_ = 0

    def set_weight(self, new_weights: list[numpy.ndarray]) -> None:
        for index, new_layer in enumerate(new_weights):
            try:
                self._weights_[index] = new_layer
            except IndexError or ValueError:
                continue

    def change_coordinates(self, x: int, y: int) -> None:
        self._rect_.x += x
        self._rect_.y += y

    def get_coordinates(self) -> tuple[int, int]:
        return self._rect_.x, self._rect_.y

    def get_rect(self) -> pygame.rect.Rect:
        return self._rect_

    def get_color(self) -> tuple[int, int, int]:
        return self._color_

    def observe(self, food_coordinates: list[tuple[int, int]]) -> numpy.ndarray:
        self_coordinates = self.get_coordinates()
        agent_x, agent_y = self_coordinates

        other_coordinates = [
            (agent_x - x, agent_y - y)
            for x in range(-OBSERVE // 2, OBSERVE // 2 + 1)
            for y in range(-OBSERVE // 2, OBSERVE // 2 + 1)
            if abs(x) > AGENT_SIZE // 2 or abs(y) > AGENT_SIZE // 2
        ]

        result = [5 if coordinate in food_coordinates else -1 for coordinate in other_coordinates]

        return numpy.array(result)

    def decision(self, food_coordinates: list[tuple[int, int]]) -> str:
        input_layer = self.observe(food_coordinates)
        first_hidden_layer = leaky_relu(input_layer.dot(self._weights_[0]) + 1)
        second_hidden_layer = leaky_relu(first_hidden_layer.dot(self._weights_[1]) + 1)
        softmax_output = softmax(second_hidden_layer)

        return DIRECTIONS[softmax_output.argmax()]

    def move(self, direction: str) -> None:
        if direction in self._moves_:
            self._moves_[direction]()

    def get_older(self) -> None:
        self._health_ -= 1

    def check_die(self) -> None:
        return self._health_ <= 0

    def get_gens_with_mutation(self) -> numpy.ndarray:
        for layer in self._weights_:
            samples = layer.shape[1:]
            layer[numpy.random.choice(len(layer), size=4, replace=False)] = numpy.random.random_sample((4, *samples))

        return self._weights_

    def get_pure_gens(self) -> numpy.ndarray:
        return self._weights_

    def eat(self) -> None:
        if self._eat_counter_ < 3:
            self._health_ += FPS // 3

            self._eat_counter_ += 1
        self._can_give_offspring_ = True

    def get_check_offspring(self) -> bool:
        return self._can_give_offspring_

    def get_eat_counter(self) -> int:
        return self._eat_counter_


class Food:
    def __init__(self, color: tuple[int, int, int] = (255, 255, 255)):
        self._color_ = color

        self._rect_ = pygame.rect.Rect(*numpy.random.randint(0, DISPLAY_SIZE), 1, 1)

    def get_rect(self) -> pygame.rect.Rect:
        return self._rect_

    def get_color(self) -> tuple[int, int, int]:
        return self._color_

    def get_coordinates(self) -> tuple[int, int]:
        return self._rect_.x, self._rect_.y

    def respawn(self) -> None:
        self._rect_.x, self._rect_.y = numpy.random.randint(0, DISPLAY_SIZE)
