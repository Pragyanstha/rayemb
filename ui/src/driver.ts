import axios from "axios";
const apiUrl = "http://localhost:8000";

export function getImage() {
    return axios.get(`${apiUrl}/image/xray`, { responseType: 'blob' })
        .then(response => {
            const imageUrl = URL.createObjectURL(response.data); // Create a URL for the blob
            return imageUrl; // Return the URL to be used in an <img> tag or similar
        });
}

export function getHeatmap(xyz: number[]) {
    const data = [{ x: xyz[0], y: xyz[1], z: xyz[2] }];
    return axios.post(`${apiUrl}/inference`, data, {
            headers: {
                'Content-Type': 'application/json', // Ensure it's sent as JSON
            },
            responseType: 'blob',
        })
        .then(response => {
            console.log("response", response);
            const heatmapUrl = URL.createObjectURL(response.data);
            return heatmapUrl;
        });
}

export function setXray(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return axios.post(`${apiUrl}/image/xray`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
}

// export function setCT(ct_path: string = "", template_path: string = "") {
//     if (ct_path === "" && template_path === "") {
//         return axios.post(`${apiUrl}/image/ct`);
//     }
//     return axios.post(`${apiUrl}/image/ct`, { ct_path: ct_path, template_path: template_path });
// }