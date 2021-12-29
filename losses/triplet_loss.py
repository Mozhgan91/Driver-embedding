# Implementation of triplet loss partly taken and modified from
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/master/losses/triplet_loss.py, #originally created
# with the following license.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import torch
import numpy


class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for represtenting the driving time series. 

    @param compared_length Maximum length of randomly chosen time series. 
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        length_pos_neg = numpy.random.randint(1, high=length + 1)

        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )

        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        ) 
        beginning_positive = beginning_batches + beginning_samples_pos
       
        end_positive = beginning_positive + length_pos_neg

        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss
