from dataloader import linear_schedule, cosine_schedule
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from torchvision.utils import make_grid


class SamplingClass():
    def __init__(self,model,batch_size,schedule,steps,img_ch,n_classes=0,device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.n_classes = n_classes
        self.device = device
        self.img_ch = img_ch
        self.schedule = schedule
        
        assert schedule == "cosine" or schedule == "linear"
        if schedule == "linear":
            data = linear_schedule(steps)
        elif schedule == "cosine":
            data = cosine_schedule(steps)
                
        self.beta_t = data['betas']
        self.alpha_t = data['alphas']
        self.alpha_t_dash = data['alphas_cumprod']
        self.sqrt_one_minus_alphas_t_dash = data['sqrt_one_minus_alphas_cumprod']
        
        self.null_idx = n_classes
    
    def process_latents(self, model, x_ts, scale_factor=1.0):
        new_x_ts = []
        for pred_latent in tqdm(x_ts, total=len(x_ts)):
            with torch.no_grad():
                pred_latent /= scale_factor
                pred_image = model.post_quant_conv(pred_latent)
                for layer in model.decoder_blocks:
                    pred_image = layer(pred_image)
                pred_image = model.proj_out(pred_image)
            
            new_x_ts.append(pred_image)
        return new_x_ts
    
    def get_classifier_score(input_image,t,target_class,classifier_model,guidance_scale):
        logits = classifier_model(input_image,t)
        prob = torch.nn.functional.softmax(logits,dim=-1)
        log_prob = torch.log(prob)    
        target_log_prob = log_prob[0,target_class]
        grad_log_prob = torch.autograd.grad(target_log_prob, input_image)[0]
        return grad_log_prob * guidance_scale

           
    def ClassifierFree(self, labels:list, weight:float):
        
        assert len(labels) >= self.batch_size
        x_t = torch.randn(size=(self.batch_size,self.img_ch,32,32),device=self.device)
        labels = torch.tensor(labels[:self.batch_size],dtype=torch.long,device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise_cond = self.model(x_t, time, labels)
                pred_score_cond = - (predicted_noise_cond / self.sqrt_one_minus_alphas_t_dash[t_idx] )
                
                null_labels = torch.tensor([self.null_idx]*self.batch_size, dtype=torch.int, device=self.device)
                predicted_noise_uncond = self.model(x_t, time, null_labels)
                pred_score_uncond = - ( predicted_noise_uncond / self.sqrt_one_minus_alphas_t_dash[t_idx] )
    
                pred_score = (1+weight) * pred_score_cond - weight * pred_score_uncond 
            
                z = torch.randn_like(predicted_noise_cond)
                if t_idx == 0:
                    z = 0

                alpha_bar_curr = self.alpha_t_dash[t_idx]
                eps = 10e-5 * (self.sqrt_one_minus_alphas_t_dash[t_idx]**2 / self.sqrt_one_minus_alphas_t_dash[0]**2)

                # eps = 2 * 10e-5 * (alpha_bar_curr)

                x_t_minus_1 = x_t + eps * pred_score + torch.sqrt(2*eps)*torch.randn_like(x_t)

                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts

    def ClassifierFree2(self, labels:list, weight:float):
        
        assert len(labels) >= self.batch_size
        x_t = torch.randn(size=(self.batch_size,self.img_ch,32,32),device=self.device)
        labels = torch.tensor(labels[:self.batch_size],dtype=torch.long,device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise_cond = self.model(x_t, time, labels)
                
                null_labels = torch.tensor([self.null_idx]*self.batch_size, dtype=torch.int, device=self.device)
                predicted_noise_uncond = self.model(x_t, time, null_labels)
    
                predicted_noise = (1+weight) * predicted_noise_cond - weight * predicted_noise_uncond 
            
                pred_x0 = (x_t - self.sqrt_one_minus_alphas_t_dash[t_idx]*predicted_noise) / torch.sqrt(self.alpha_t_dash[t_idx])
                pred_x0 = pred_x0.clamp(-1.0,1.0)
                
                a_t = self.alpha_t[t_idx]
                a_dash_t = self.alpha_t_dash[t_idx]
                a_dash_t_minus_one = self.alpha_t_dash[t_idx - 1] if t_idx > 0 else torch.tensor(1.0).to(self.device)

                c1 = (torch.sqrt(a_t) * (1-a_dash_t_minus_one)) / (1-a_dash_t)
                c2 = (torch.sqrt(a_dash_t_minus_one)*(1-a_t)) / (1-a_dash_t)
                mean = c1*x_t + c2*pred_x0
                
                var = ((1-a_dash_t_minus_one)*(1-a_t))/(1-a_dash_t)
                
                z = torch.randn_like(predicted_noise)
                if t_idx == 0:
                    z = 0

                x_t_minus_1 = (mean + torch.sqrt(var)*z)
                
                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts
    
    def Ancestral(self, labels:list, weight:float):
        
        assert len(labels) >= self.batch_size
        x_t = torch.randn(size=(self.batch_size,self.img_ch,32,32),device=self.device)
        labels = torch.tensor(labels[:self.batch_size],dtype=torch.long,device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time, labels)
            
                z = torch.randn_like(predicted_noise)
                if t_idx == 0:
                    z = 0

                a_t = self.alpha_t[t_idx]
                a_dash_t = self.alpha_t_dash[t_idx]
                a_dash_t_minus_one = self.alpha_t_dash[t_idx-1]

                mean =  (1/torch.sqrt(a_t))* (x_t - ((1 - a_t) / torch.sqrt(1 - a_dash_t)) * predicted_noise)
                var = ((1-a_dash_t_minus_one)*(1-a_t))/(1-a_dash_t)
                
                x_t_minus_1 = (mean + torch.sqrt(var) * z)
                
                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts

    def Ancestral2(self):
        
        x_t = torch.randn(size=(self.batch_size,self.img_ch,16, 16),device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time)

                # x_t = sqrt_alpha_dah_t * x_0 + sqrt_1_minus_alpha_dash_t * z
                # x_0 = ( x_t - sqrt_1_minus_alpha_dash_t * z ) / sqrt_alpha_dah_t
                
                pred_x0 = (x_t - self.sqrt_one_minus_alphas_t_dash[t_idx]*predicted_noise) / torch.sqrt(self.alpha_t_dash[t_idx])
                pred_x0 = pred_x0.clamp(-1.0,1.0)
                
                a_t = self.alpha_t[t_idx]
                a_dash_t = self.alpha_t_dash[t_idx]
                a_dash_t_minus_one = self.alpha_t_dash[t_idx - 1] if t_idx > 0 else torch.tensor(1.0).to(self.device)

                c1 = (torch.sqrt(a_t) * (1-a_dash_t_minus_one)) / (1-a_dash_t)
                c2 = (torch.sqrt(a_dash_t_minus_one)*(1-a_t)) / (1-a_dash_t)
                mean = c1*x_t + c2*pred_x0
                
                var = ((1-a_dash_t_minus_one)*(1-a_t))/(1-a_dash_t)
                
                z = torch.randn_like(predicted_noise)
                if t_idx < 5:
                    z = 0

                x_t_minus_1 = (mean + torch.sqrt(var)*z)
                
                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts

    def LangevinDynamics(self, labels:list, weight:float):
        
        assert len(labels) >= self.batch_size
        x_t = torch.randn(size=(self.batch_size,self.img_ch,32,32),device=self.device)
        labels = torch.tensor(labels[:self.batch_size],dtype=torch.long,device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time, labels)
                pred_score = - (predicted_noise / self.sqrt_one_minus_alphas_t_dash[t_idx] )
                
            
                z = torch.randn_like(predicted_noise)
                if t_idx == 0:
                    z = 0

                eps = 10e-5 * (self.sqrt_one_minus_alphas_t_dash[t_idx]**2 / self.sqrt_one_minus_alphas_t_dash[0]**2)

                x_t_minus_1 = x_t + eps * pred_score + torch.sqrt(2*eps)*torch.randn_like(x_t)

                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts

    def LangevinDynamics2(self, labels: list, weight: float):
        
        assert len(labels) >= self.batch_size
        # Start from pure noise
        x_t = torch.randn(size=(self.batch_size, self.img_ch, 32, 32), device=self.device)
        labels = torch.tensor(labels[:self.batch_size], dtype=torch.long, device=self.device)

        x_ts = [x_t]
        
        # Hyperparameters
        # 2e-6 is a safe starting point for CIFAR-10 with standard schedules
        target_snr = 0.16 
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0, self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,), fill_value=t_idx, dtype=torch.long, device=self.device)
                
                # 1. Predict Noise and Score
                predicted_noise = self.model(x_t, time, labels)
                
                # Get current noise level (sigma_t)
                sigma_t = self.sqrt_one_minus_alphas_t_dash[t_idx]
                
                # Score = -Noise / Sigma
                pred_score = - (predicted_noise / sigma_t)

                # 2. Dynamic Step Size Calculation (SNR-based)
                # This prevents explosion by ensuring step size scales with the noise level
                noise_norm = torch.norm(torch.randn_like(x_t).reshape(x_t.shape[0], -1), dim=-1).mean()
                grad_norm = torch.norm(pred_score.reshape(pred_score.shape[0], -1), dim=-1).mean()
                
                # eps = (SNR * ||noise|| / ||score||)^2 * 2 * sigma_t^2
                # We scale by 2 * alpha (here approximated via sigma_t relative change)
                # For pure Langevin, a robust heuristic is:
                eps = 2 * (target_snr * noise_norm / grad_norm) ** 2

                # 3. Update Step
                z = torch.randn_like(predicted_noise)
                
                # CRITICAL FIX: Do not add noise at the final step (t=0)
                if t_idx == 0:
                    z = 0
                    noise_term = 0
                else:
                    noise_term = torch.sqrt(2 * eps) * z

                # Langevin Update: x_{t-1} = x_t + eps * score + sqrt(2*eps) * z
                x_t_minus_1 = x_t + eps * pred_score + noise_term

                x_t = x_t_minus_1
                x_ts.append(x_t)
        
        return x_ts

    def PredictorCorrector(self, labels:list, nc:int):
        
        assert len(labels) >= self.batch_size
        x_t = torch.randn(size=(self.batch_size,self.img_ch,32,32),device=self.device)
        labels = torch.tensor(labels[:self.batch_size],dtype=torch.long,device=self.device)

        x_ts = [x_t]
        
        with torch.no_grad():
            for t_idx in tqdm(reversed(range(0,self.steps)), total=self.steps):
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time, labels)

                # x_t = sqrt_alpha_dah_t * x_0 + sqrt_1_minus_alpha_dash_t * z
                # x_0 = ( x_t - sqrt_1_minus_alpha_dash_t * z ) / sqrt_alpha_dah_t
                
                pred_x0 = (x_t - self.sqrt_one_minus_alphas_t_dash[t_idx]*predicted_noise) / torch.sqrt(self.alpha_t_dash[t_idx])
                pred_x0 = pred_x0.clamp(-1.0,1.0)
                
                a_t = self.alpha_t[t_idx]
                a_dash_t = self.alpha_t_dash[t_idx]
                a_dash_t_minus_one = self.alpha_t_dash[t_idx - 1] if t_idx > 0 else torch.tensor(1.0).to(self.device)

                c1 = (torch.sqrt(a_t) * (1-a_dash_t_minus_one)) / (1-a_dash_t)
                c2 = (torch.sqrt(a_dash_t_minus_one)*(1-a_t)) / (1-a_dash_t)
                mean = c1*x_t + c2*pred_x0
                
                var = ((1-a_dash_t_minus_one)*(1-a_t))/(1-a_dash_t)
                
                z = torch.randn_like(predicted_noise)
                if t_idx == 0:
                    z = 0

                x_t_minus_1 = (mean + torch.sqrt(var)*z)
                
                x_t = x_t_minus_1
                
                if t_idx > 0:
                    for _ in range(nc):
                        time = torch.full(size=(self.batch_size,),fill_value=t_idx-1, dtype=torch.long, device=self.device)

                        predicted_noise = self.model(x_t, time, labels)
                        pred_score = - (predicted_noise / self.sqrt_one_minus_alphas_t_dash[t_idx] )
                        
                        z = torch.randn_like(predicted_noise)
                        if t_idx == 0:
                            z = 0

                        alpha_bar_curr = self.alpha_t_dash[t_idx]
                        eps = 2 * 10e-5 * (alpha_bar_curr)

                        x_t_minus_1 = x_t + eps * pred_score + torch.sqrt(2*eps)*torch.randn_like(x_t)

                        x_t = x_t_minus_1
        
                x_ts.append(x_t)
        
        return x_ts
    
    def DDIM(self, time_steps):
        
        x_t = torch.randn(size=(self.batch_size,self.img_ch,16,16),device=self.device)

        x_ts = [x_t]
        
        iterator = tqdm(range(0,len(time_steps)), total=len(time_steps))
        with torch.no_grad():
            for i in iterator:
                
                i = len(time_steps) - 1 - i
                time_step = time_steps[i]
                t_idx = time_step - 1
                
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time)

                # x_t = sqrt_alpha_dah_t * x_0 + sqrt_1_minus_alpha_dash_t * z
                # x_0 = ( x_t - sqrt_1_minus_alpha_dash_t * z ) / sqrt_alpha_dah_t
                
                pred_x0 = (x_t - self.sqrt_one_minus_alphas_t_dash[t_idx]*predicted_noise) / torch.sqrt(self.alpha_t_dash[t_idx])
                pred_x0 = pred_x0.clamp(-1.0,1.0)
                
                prev_time_step = time_steps[i-1]
                prev_t_idx = prev_time_step - 1
                
                a_bar_new_t = self.alpha_t_dash[prev_t_idx] if time_step > 1 else torch.tensor(1.0).to(self.device)

                c1 = torch.sqrt(a_bar_new_t)
                c2 = torch.sqrt(1-a_bar_new_t)
                
                
                x_new_t = c1 * pred_x0 + c2 * predicted_noise
                
                x_t = x_new_t
                x_ts.append(x_t)
                
                iterator.set_postfix({
                    "c1" : c1.item(),
                    "c2" : c2.item(),
                    "t" : time_step
                    })
        
        return x_ts 
    
    def slerp(self, z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )

    def DDIM_Interpolation(self, time_steps):
        
        print((self.batch_size,),self.n_classes)
        labels = torch.full(size=(self.batch_size,),fill_value=10,dtype=torch.long,device=self.device)
        
        assert len(labels) >= self.batch_size
        
        z1 = torch.randn(size=(1,self.img_ch,32,32),device=self.device)
        z2 = torch.randn(size=(1,self.img_ch,32,32),device=self.device)
        alphas = torch.arange(0.0, 1.00, 1.0/self.batch_size).to(z1.device)
        
        z = []
        for i in range(0,self.batch_size):
            z.append(self.slerp(z1,z2,alphas[i]))
        
        x_t = torch.concat(z,dim=0)

        x_ts = [x_t]
        
        iterator = tqdm(range(0,len(time_steps)), total=len(time_steps))
        with torch.no_grad():
            for i in iterator:
                
                i = len(time_steps) - 1 - i
                time_step = time_steps[i]
                t_idx = time_step - 1
                
                time = torch.full(size=(self.batch_size,),fill_value=t_idx, dtype=torch.long, device=self.device)
                
                predicted_noise = self.model(x_t, time, labels)

                # x_t = sqrt_alpha_dah_t * x_0 + sqrt_1_minus_alpha_dash_t * z
                # x_0 = ( x_t - sqrt_1_minus_alpha_dash_t * z ) / sqrt_alpha_dah_t
                
                pred_x0 = (x_t - self.sqrt_one_minus_alphas_t_dash[t_idx]*predicted_noise) / torch.sqrt(self.alpha_t_dash[t_idx])
                pred_x0 = pred_x0.clamp(-1.0,1.0)
                
                prev_time_step = time_steps[i-1]
                prev_t_idx = prev_time_step - 1
                
                a_bar_new_t = self.alpha_t_dash[prev_t_idx] if time_step > 1 else torch.tensor(1.0).to(self.device)

                c1 = torch.sqrt(a_bar_new_t)
                c2 = torch.sqrt(1-a_bar_new_t)
                
                
                x_new_t = c1 * pred_x0 + c2 * predicted_noise
                
                x_t = x_new_t
                x_ts.append(x_t)
                
                iterator.set_postfix({
                    "c1" : c1.item(),
                    "c2" : c2.item(),
                    "t" : time_step
                    })
        
        return x_ts    
    
    def visualize(self, x_ts):
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            # 1. Process the tensors
            # Stack list of tensors into (Steps, Batch, Channels, H, W)
            stacked = torch.stack(x_ts, dim=0)
            
            # Normalize: (x + 1) / 2 maps [-1, 1] to [0, 1]
            # (Your original code added 0.5 without scaling, which results in [-0.5, 1.5])
            stacked = (stacked.clamp(-1, 1) + 1) / 2 
            
            # Permute to (Steps, Batch, H, W, Channels) for matplotlib
            stacked = stacked.permute(0, 1, 3, 4, 2).cpu().numpy()
            
            # 2. Organize frames by batch item
            # frames[i] will contain the timeline for the ith image in the batch
            frames = [[] for _ in range(self.batch_size)]
            
            total_steps = stacked.shape[0]
            
            for t in range(total_steps):
                for b in range(self.batch_size):
                    # Append the image at step t for batch item b
                    frames[b].append(stacked[t, b])

            # 3. Animation Section
            # Handle single batch size edge case for subplots
            if self.batch_size == 1:
                fig, axes = plt.subplots(1, 1, figsize=(4, 4))
                axes = [axes] # Wrap in list to make iterable
            else:
                fig, axes = plt.subplots(1, self.batch_size, figsize=(self.batch_size * 3, 3))
                
            plt.suptitle(f"Denoising Process ({str(self.schedule).capitalize()} Schedule)")

            # Initialize images
            ims = []
            for i in range(self.batch_size):
                axes[i].axis("off")
                axes[i].set_title(f"Sample {i}")
                # Initialize with the first frame (pure noise)
                im = axes[i].imshow(frames[i][0])
                ims.append(im)

            # Update function for animation
            def update(frame_idx):
                for i in range(self.batch_size):
                    ims[i].set_data(frames[i][frame_idx])
                return ims

            # Create animation
            ani = animation.FuncAnimation(
                fig, 
                update, 
                frames=len(frames[0]), 
                interval=5, # 5ms per frame
                blit=True
            )
            
            plt.tight_layout()
            plt.show()
            
            final_images = [frames[i][-1] for i in range(self.batch_size)]

            fig_final, axes_final = plt.subplots(1, self.batch_size, figsize=(self.batch_size * 3, 3))
            if self.batch_size == 1:
                axes_final = [axes_final]

            for i, ax in enumerate(axes_final):
                img = final_images[i]
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img, vmin=0, vmax=1)
                ax.set_title(f"Sample {i+1}")
                ax.axis("off")

            plt.suptitle("Final Generated Images (x₀)", fontsize=14)
            plt.tight_layout()
            plt.show()      

    def visualize_stats(self, x_ts, sampling_strategy):
        """
        Plots Mean, Std, Min, and Max trajectories over the denoising steps.
        Includes text annotations for final values.
        """
        import torch
        import matplotlib.pyplot as plt
        
        # 1. Process
        stacked = torch.stack(x_ts).cpu()
        # Flatten: (Steps, Batch * Channels * H * W)
        flat_data = stacked.view(stacked.shape[0], -1)
        
        # 2. Calculate Stats
        means = flat_data.mean(dim=1)
        stds = flat_data.std(dim=1)
        mins = flat_data.min(dim=1).values
        maxs = flat_data.max(dim=1).values
        
        steps = range(len(means))

        # 3. Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Plot 1: Value Range (Min, Max, Mean) ---
        # Plot the envelope
        axes[0].plot(steps, maxs, color='#2ca02c', linestyle=':', alpha=0.6, label='Max')
        axes[0].plot(steps, mins, color='#d62728', linestyle=':', alpha=0.6, label='Min')
        # Plot the mean
        axes[0].plot(steps, means, color='#1f77b4', linewidth=2, label='Mean')
        
        # Fill area between min and max for better visualization
        axes[0].fill_between(steps, mins, maxs, color='gray', alpha=0.1)
        
        axes[0].set_title("Pixel Value Range (Min/Max/Mean)")
        axes[0].set_xlabel("Reverse Process Step")
        axes[0].set_ylabel("Pixel Value")
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Annotate Final Values on Plot 1
        text_str = (f"Final Step:\n"
                    f"Max: {maxs[-1]:.2f}\n"
                    f"Mean: {means[-1]:.2f}\n"
                    f"Min: {mins[-1]:.2f}")
        # Place text box in top right
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        axes[0].text(0.95, 0.95, text_str, transform=axes[0].transAxes, verticalalignment='top', horizontalalignment='right', bbox=props)

        # --- Plot 2: Standard Deviation ---
        axes[1].plot(steps, stds, color='#ff7f0e', linewidth=2, label='Std Dev')
        axes[1].set_title("Standard Deviation Trajectory")
        axes[1].set_xlabel("Reverse Process Step")
        axes[1].set_ylabel("Std Dev")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label="Unit Variance")
        axes[1].legend()

        # Annotate Final Std on Plot 2
        axes[1].text(0.95, 0.95, f"Final Std: {stds[-1]:.4f}", transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.suptitle(f"Denoising Statistics ({str(self.schedule).capitalize()} Schedule) | {sampling_strategy}", fontsize=14)
        plt.tight_layout()
        plt.show()         