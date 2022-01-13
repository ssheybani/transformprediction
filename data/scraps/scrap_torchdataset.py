class movingToysTorchDataset1(torch.utils.data.Dataset):
    """
    methods:
    
    container for the samples
    selection:
        by index and slice:
            __getitem__
            __len__
    
    sampling:
        
        random sample
    
    Not immediately needed:
    plotting samples:
        show_samples()
    """

    def __init__(self, movingToysDataset, target_latent=None, 
                 torchvision_transforms=None, resize=None, rgb_expand=False, cielab=False):
        """
        Initialized a custom Torch dataset for dSprites, and sets attributes.

        NOTE: Always check that transforms behave as expected (e.g., produce   
        outputs in expected range), as datatypes (e.g., torch vs numpy, 
        uint8 vs float32) can change the behaviours of certain transforms, 
        e.g. ToPILImage.

        Required args:
        - dSprites (dSpritesDataset): dSprites dataset

        Optional args:
        - target_latent (str): latent dimension to use as target. 
            (default: "shape")
        - torchvision_transforms (torchvision.transforms): torchvision 
            transforms to apply to X. (default: None)
        - resize (None or int): if not None, should be an int, namely the 
            size to which X is expanded along its height and width. 
            (default: None)
        - rgb_expand (bool): if True, X is expanded to include 3 identical 
            channels. Applied after any torchvision_tranforms. 
            (default: False)
        - simclr (bool or str): if True, SimCLR-specific transformations are 
            applied. (default: False)
        - simclr_mode (str): If not None, determines whether data is returned 
            in 'train' mode (with augmentations) or 'test' mode (no augmentations). 
            Ignored if simclr is False. 
            (default: 'train') 
        - simclr_transforms (torchvision.transforms): SimCLR-specific 
            transforms. If "spijk", then SimCLR transforms from (https://github.com/Spijkervet/SimCLR), 
            are ised. If None, default SimCLR transforms are applied. Ignored if 
            simclr is False. (default: None)

        Sets attributes:
        - X (2 or 3D np array): image array 
            (channels (optional) x height x width).
        - y (1D np array): targets

        ...
        """

        self.dSprites = dSprites
        self.target_latent = target_latent

        self.X = self.dSprites.images
        self.y = self.dSprites.get_latent_classes(
            latent_class_names=target_latent
            ).squeeze()
        self.num_classes = \
            len(self.dSprites.latent_class_values[self.target_latent])
        
        if len(self.X) != len(self.y):
            raise ValueError(
                "images and latent classes must have the same length, but "
                f"found {len(self.X)} and {len(self.y)}, respectively."
                )
        
        if len(self.X.shape) not in [3, 4]:
            raise ValueError("images should have 3 or 4 dimensions, but "
                f"found {len(self.X.shape)}.")

        self.simclr = simclr
        self.simclr_mode = None
        self.simclr_transforms = None
        if self.simclr:
            self.simclr_mode = simclr_mode
            self.spijk = (simclr_transforms == "spijk")
            if self.simclr_mode not in ["train", "test"]:
                raise ValueError("simclr_mode must be 'train' or 'test', but "
                    f"found {self.simclr_mode}.")

            if self.spijk:
                torchvision_transforms = False
                if len(self.X[0].shape) == 2:
                    rgb_expand = True

                from simclr.modules.transformations import TransformsSimCLR
                if self.simclr_mode == "train":
                    self.simclr_transforms = \
                        TransformsSimCLR(size=224).train_transform
                else:
                    self.simclr_transforms = \
                        TransformsSimCLR(size=224).test_transform

            else:
                if self.simclr_mode == "train":
                    self.simclr_transforms = simclr_transforms
                    if self.simclr_transforms is None:
                        self.simclr_transforms = \
                            torchvision.transforms.RandomAffine(
                                degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)
                            )
                else:
                    self.simclr_transforms = None

        self.torchvision_transforms = torchvision_transforms
        
        self.resize = resize
        if self.resize is not None:
            self.resize_transform = \
                torchvision.transforms.Resize(size=self.resize)

        self.rgb_expand = rgb_expand
        if self.rgb_expand and len(self.X[0].shape) != 2:
            raise ValueError(
                "If rgb_expand is True, X should have 2 dimensions, but it"
                f" has {len(self.X[0].shape)} dimensions."
                )

        self._ch_expand = False
        if len(self.X[0].shape) == 2 and not self.rgb_expand:
            self._ch_expand = True

        self.num_samples = len(self.X)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        y = self.y[idx]

        if self.rgb_expand:
            X = np.repeat(np.expand_dims(X, axis=-3), 3, axis=-3)

        if self._ch_expand:
            X = np.expand_dims(X, axis=-3)

        X = torch.tensor(X)


        if self.simclr and self.spijk:
            X = self._preprocess_simclr_spijk(X)
        else:
            if self.resize is not None:
                X = self.resize_transform(X)

            if self.torchvision_transforms is not None:
                X = self.torchvision_transforms()(X)

        y = torch.tensor(y)

        if self.simclr:
            if self.simclr_transforms is None: # e.g. in test mode
                X_aug1, X_aug2 = X, X
            else:
                X_aug1 = self.simclr_transforms(X)
                X_aug2 = self.simclr_transforms(X) 
            return (X_aug1, X_aug2, y, idx)
        else:
            return (X, y, idx)

    def _preprocess_simclr_spijk(self, X):
        """
        self._preprocess_simclr_spijk(X)
        
        Preprocess X for SimCLR transformations of the SimCLR implementation 
        available here: https://github.com/Spijkervet/SimCLR

        Required args:
        - X (2 or 3D np array): image array 
            (height x width x channels (optional)). 
            All values expected to be between 0 and 1.
        
        Returns:
        - X (3 or 4D np array): image array 
                                ((images) x height x width x channels).
        """

        if X.max() > 1 or X.min() < 0:
            raise NotImplementedError(
                "Expected X to be between 0 and 1 for SimCLR transform."
                )

        if len(X.shape) == 4:
            raise NotImplementedError(
                "Slicing dataset with multiple index values at once not "
                "supported, due to use of PIL torchvision transforms."
                )
        
        # input must be torch Tensor to be correctly interpreted
        X = torchvision.transforms.ToPILImage(mode="RGB")(X)

        return X


    def show_images(self, indices=None, num_images=10, ncols=5, randst=None, 
                    annotations=None):
        """
        self.show_images()

        Plots dSprites images, or their augmentations if applicable.

        Optional args:
        - indices (array-like): indices of images to plot. If None, they are 
            sampled randomly. (default: None)
        - num_images (int): number of images to sample and plot, if indices is 
            None. (default: 10)
        - ncols (int): number of columns to plot. (default: 5)
        - randst (np.random.RandomState): seed or random state to use if 
            sampling images. If None, the global state is used. (Does not 
            control SimCLR transformations.) (default: None)
        - annotations (str): If not None, annotations are added to images, 
            e.g., 'posX_quadrants'. (default: None)
        """

        if indices is None:
            if num_images > self.num_samples:
                raise ValueError("Cannot sample more images than the number "
                    f"of images in the dataset ({self.num_samples}).")
            if randst is None:
                randst = np.random
            elif isinstance(randst, int):
                randst = np.random.RandomState(randst)
            indices = randst.choice(
                np.arange(self.num_samples), num_images, replace=False
                )
        else:
            num_images = len(indices)

        centers = None
        if annotations is not None:
            if self.simclr and self.simclr_mode == "train":
                # all data is augmented, so centers cannot be identified
                centers = None
            else:
                centers = self.dSprites.get_latent_values(
                    indices, latent_class_names=["posX", "posY"]
                    ).tolist()

        Xs, X_augs1, X_augs2 = [], [], []
        for idx in indices:
            if self.simclr:
                X_aug1, X_aug2, _, _ = self[idx]
                X_augs1.append(X_aug1.numpy())
                X_augs2.append(X_aug2.numpy())
            else:
                X, _, _ = self[idx]
                Xs.append(X.numpy())

        
        if self.simclr:
            title = f"{num_images} pairs of dataset image augmentations"
            fig, _ = plot_util.plot_dsprite_image_doubles(
                X_augs1, X_augs2, ["Augm. 1", "Augm. 2"], ncols=ncols, annotations=annotations, 
                centers=[centers, None]
                )
        else:
            title = f"{num_images} dataset images"
            fig, _ = plot_util.plot_dsprites_images(
                Xs, ncols=ncols, annotations=annotations, centers=centers
                )
        
        y = 1.04
        if annotations is not None:
            title = f"{title}\nwith annotations (red)"
            y = 1.1

        fig.suptitle(title, y=1.04)
