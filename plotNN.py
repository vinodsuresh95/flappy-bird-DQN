#This file is used to plot the neural network
#the first and last layers are accurate, middle ones are just for visualization
import pygame
import math
import random
import threading
import os

class PygameWindow:
    def __init__(self, window_title="Pygame Window", width=800, height=600):
        self.window_title = window_title
        self.width = width
        self.height = height
        self.running = False
        self.thread = None
        self.layer_sizes = [12, 25, 25, 2]
        self.node_radius = 11
        self.nodes = []
        self.synapses = []
        self.previous_inputs = None
        self.output_value = 0 
        self.color_update_counter = 0 
        self.color_update_frequency = 14  
        self.middle_layer_colors = [None] * 2  

        
        self.initialize_middle_layer_colors()

    def initialize_middle_layer_colors(self):
        self.middle_layer_colors = []
        for layer_idx in range(2):
            layer_colors = [(255, 255, 255)] * self.layer_sizes[layer_idx + 1]  
            self.middle_layer_colors.append(layer_colors)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def initialize_synapses(self):
        max_layer_size = max(self.layer_sizes)
        layer_gap = self.width // (len(self.layer_sizes) + 1)
        
       
        node_gap = 2 * self.node_radius

        nodes = []
        for i, layer_size in enumerate(self.layer_sizes):
            layer_nodes = []
            for j in range(layer_size):
                x = (i + 1) * layer_gap
                y = (j + 1) * node_gap + (self.height - layer_size * node_gap) / 2
                layer_nodes.append((x, y))
            nodes.append(layer_nodes)
        
        
        synapses = []
        for i in range(len(nodes) - 1):
            for node1 in nodes[i]:
                for node2 in nodes[i + 1]:
                    angle = math.atan2(node2[1] - node1[1], node2[0] - node1[0])
                    start_x = node1[0] + self.node_radius * math.cos(angle)
                    start_y = node1[1] + self.node_radius * math.sin(angle)
                    end_x = node2[0] - self.node_radius * math.cos(angle)
                    end_y = node2[1] - self.node_radius * math.sin(angle)

                    
                    color = random.choice([(139, 0, 139), (0, 0, 139)])  
                    synapses.append(((start_x, start_y), (end_x, end_y), color))
        
        self.nodes = nodes
        self.synapses = synapses

    def read_inputs(self):
        
        if os.path.exists('input.txt') and os.path.getsize('input.txt') > 0:
            with open('input.txt', 'r') as file:
                inputs = list(map(float, file.read().strip().split()))

           
            if len(inputs) == 12:
                
                normalized_inputs = [(value + 1) / 2 for value in inputs]
            else:
                normalized_inputs = [0.5] * 12 
        else:
            normalized_inputs = [0.5] * 12 

        
        if self.previous_inputs is None or normalized_inputs != self.previous_inputs:
            self.previous_inputs = normalized_inputs
            
            if self.color_update_counter >= self.color_update_frequency:
                self.update_middle_layer_colors()
                self.color_update_counter = 0 
            else:
                self.color_update_counter += 1

        return normalized_inputs

    def update_middle_layer_colors(self):
        
        self.middle_layer_colors = []
        for layer_idx in range(2):
            layer_colors = []
            selected_nodes = random.sample(range(self.layer_sizes[layer_idx + 1]), 6) 
            for node_idx in range(self.layer_sizes[layer_idx + 1]):  
                if node_idx in selected_nodes:
                    
                    green_intensity = random.randint(100, 200)
                    color = (0, green_intensity, 0)
                else:
                    color = (255, 255, 255) 
                layer_colors.append(color)
            self.middle_layer_colors.append(layer_colors)

    def read_output(self):
        
        if os.path.exists('output.txt') and os.path.getsize('output.txt') > 0:
            with open('output.txt', 'r') as file:
                try:
                    self.output_value = int(file.read().strip())
                except ValueError:
                    self.output_value = 0 
        else:
            self.output_value = 0

    def draw_mlp(self, screen):
       
        normalized_inputs = self.read_inputs()
        
        self.read_output()

        
        for (start, end, color) in self.synapses:
            pygame.draw.line(screen, color, start, end, 2)

        
        for layer_idx, layer_nodes in enumerate(self.nodes):
            for node_idx, (x, y) in enumerate(layer_nodes):
                if layer_idx == 0:  
                    
                    green_intensity = int(abs(normalized_inputs[node_idx]) * 255)
                    
                    color = (255 - green_intensity, 255, 255 - green_intensity)
                elif layer_idx in [1, 2]: 
                    
                    color = self.middle_layer_colors[layer_idx - 1][node_idx]
                elif layer_idx == 3:  
                    if (self.output_value == 1 and node_idx == 0) or (self.output_value != 1 and node_idx == 1):
                        color = (0, 255, 0)  
                    else:
                        color = (255, 255, 255) 
                else:
                    color = (255, 255, 255)

                pygame.draw.circle(screen, color, (x, y), self.node_radius)
                pygame.draw.circle(screen, (0, 0, 0), (x, y), self.node_radius, 2) 

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.window_title)

        self.initialize_synapses()

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            screen.fill((0, 0, 0))  
            self.draw_mlp(screen)
            pygame.display.flip()

            clock.tick(30)  
        
        pygame.quit()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()

window = PygameWindow(window_title="MLP Visualization")

window.start()


