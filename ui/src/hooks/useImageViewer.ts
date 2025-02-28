import { useState } from 'react';
import { setXray, getImage } from '../driver';

const useImageViewer = (setImageUrl: (url: string) => void) => {
    const [isUploading, setIsUploading] = useState(false);

    const uploadImage = async (file: File) => {
        setIsUploading(true);
        try {
            // Handle successful upload
            console.log('Image upload');
            console.log(file);
            setXray(file);
            getImage().then(setImageUrl);
        } catch (error) {
            console.error('Error uploading image:', error);
        } finally {
            setIsUploading(false);
        }
    };

    return { isUploading, uploadImage };
};

export default useImageViewer; 