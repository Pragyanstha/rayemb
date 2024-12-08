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

// Not used until I figure out how to manage sessions per user
// export function setXray(path: string = "") {
//     if (path === "") {
//         return axios.post(`${apiUrl}/image/xray`);
//     }
//     return axios.post(`${apiUrl}/image/xray`, { image_path: path });
// }

// export function setCT(ct_path: string = "", template_path: string = "") {
//     if (ct_path === "" && template_path === "") {
//         return axios.post(`${apiUrl}/image/ct`);
//     }
//     return axios.post(`${apiUrl}/image/ct`, { ct_path: ct_path, template_path: template_path });
// }