import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getInfo        = ()              => api.get('/info').then(r => r.data)
export const getBatch       = (idx)           => api.get(`/batch/${idx}`).then(r => r.data)
export const getImageUrl    = (id)            => `/api/image/${id}`
export const postEdit       = (annotation_id, new_category_id, attributes) =>
  api.post('/edit',       { annotation_id, new_category_id, attributes }).then(r => r.data)
export const postDelete     = (annotation_id) =>
  api.post('/delete',     { annotation_id }).then(r => r.data)
export const postUndoDelete = (annotation_id) =>
  api.post('/undo_delete',{ annotation_id }).then(r => r.data)
export const postAdd        = (image_id, segmentation, category_id, attributes) =>
  api.post('/add',        { image_id, segmentation, category_id, attributes }).then(r => r.data)
export const postApprove    = (idx)           => api.post(`/approve/${idx}`).then(r => r.data)
export const postUnapprove  = (idx)           => api.post(`/unapprove/${idx}`).then(r => r.data)
