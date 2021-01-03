
class VideoPredictor():
    """
    A forward pass for a full video in a single pass is a memory heavy operation. For systems
    which run out of memory during this process, be it RAM or VRAM, this class provides
    a solution. This class is a wrapper that splits the video into batches of sequential
    frames and makes a forward pass for each batch. The output is the concatenated to a list
    of predictions for each batch. 
    
    On some occations, the network architecture has an uncovenient output to contatenate
    into a list. A user can provide an output handler to fix these outputs into a preferable
    format. 
    """
    def __init__(self, model, no_batches, device, output_handler=None):
        self.model = model
        self.no_batches = no_batches
        self.device = device
        self.output_handler = output_handler


    def predict(self, sample):
        data = sample.to(self.device)

        no_frames = data.shape[0]
        batch_size = int(no_frames/self.no_batches)

        outputs = []
        for i in range(self.no_batches+1):
            if i == self.no_batches: 
                # Once the final batch is reached, the rest of the frames are predicted
                batch = data[i*batch_size:]
            else:
                batch = data[i*batch_size:(i+1)*batch_size]
            
            output = self.model(batch)
            del batch
            outputs.append(output) # TODO: Find out if tensors are cast to cpu

        if self.output_handler:
            return self.output_handler(outputs)
        else:
            return outputs
